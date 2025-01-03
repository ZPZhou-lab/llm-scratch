from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.RESIDUAL_SCALE_INIT = 1.0
        self.gelu = nn.GELU(approximate='tanh')
    
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections in one branch when doing multihead attention
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALE_INIT = 1.0
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # shape: (1, 1, config.block_size, config.block_size)
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # (batch, tokens, embed_size)
        qkv = self.c_attn(x) # (batch, tokens, 3 * embed_size)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (batch, n_head, tokens, embed_size // n_head)
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)

        # # calculate attention scores （batch, n_head, tokens, tokens)
        # attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        # attn = F.softmax(attn, dim=-1)
        # # apply attention to the values
        # y = attn @ v # (batch, n_head, tokens, embed_size // n_head)
        # use flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        # output projection
        return self.c_proj(y)


class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        # residual connection
        x = x + self.attn(self.ln_1(x)) 
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing for wte
        self.transformer.wte.weight = self.lm_head.weight

        # initialize weights
        self.apply(self._init_weights)
    
    def build_optimizer(self, lr: float, weight_decay: float=0.1, fused: bool=True, **kwargs):
        # get parameters
        params = [param for name, param in self.named_parameters()]
        # remove bias and layernorm weights, remove params without gradients
        dec_params = [param for param in params if len(param.size()) > 1 and param.requires_grad]
        non_dec_params = [param for param in params if len(param.size()) <= 1 and param.requires_grad]
        # print number of parameters
        total_dec_params = sum(param.numel() for param in dec_params)
        total_non_dec_params = sum(param.numel() for param in non_dec_params)
        print(f"Totol parameters use weight-decay: {total_dec_params}")
        print(f"Totol parameters without weight-decay: {total_non_dec_params}")

        # build optimizer
        optimizer = torch.optim.AdamW(
            dec_params, lr=lr, weight_decay=weight_decay, fused=fused,
            betas=kwargs.get('betas', (0.9, 0.95)),
            eps=kwargs.get('eps', 1e-8)
        )
        return optimizer

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'RESIDUAL_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, inputs, targets=None):
        B, T = inputs.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        # tokens embeddings
        pos = torch.arange(T, dtype=torch.long, device=inputs.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(inputs)
        x = tok_emb + pos_emb
        # transformer blocks
        for block in self.transformer.h:
            x = block(x)
        # forward pass of the final layer norm
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (batch, tokens, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    
    def load_from_state_dict(self, state_dict):
        sd = self.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.mask')]

        # copy state dict from given
        sd_keys_hf = state_dict.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys) == len(sd_keys_hf), f"Length mismatch: {len(sd_keys)} vs {len(sd_keys_hf)}"
        for k in sd_keys_hf:
            if any(k.endswith(t) for t in transposed):
                assert state_dict[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(state_dict[k].t())
            else:
                assert state_dict[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(state_dict[k])