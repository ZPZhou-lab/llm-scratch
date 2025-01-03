import torch
import numpy as np
import math
import time
import tiktoken
from gpt2 import GPT, GPT2Config
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import os

from helloswag_eval import evaluate_ddp

class DataLoaeder:
    def __init__(self, 
        batch_size: int, 
        seq_len: int, 
        ddp_world_size: int,
        rank: int,
        split: str='train'
    ) -> None:
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.ddp_world_size = ddp_world_size
        self.rank = rank
        assert split in ['train', 'valid']
        self.split = split

        # load data from path and prepare shards
        data_root = 'fineweb_edu_10B'
        shards = os.listdir(data_root)
        shards = [shard for shard in shards if split in shard]
        shards = sorted(shards)
        shards = [os.path.join(data_root, shard) for shard in shards]
        self.shards = shards
        assert len(self.shards) > 0
        if rank == 0:
            print(f"load total {len(self.shards)} shards")
        
        # init shard index
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_pos = self.rank * self.batch_size * self.seq_len
    
    def next_batch(self):
        B, T = self.batch_size, self.seq_len
        buf = self.tokens[self.current_pos : self.current_pos + B * T + 1]
        # create inputs and targets
        inputs  = buf[:-1].view(B, T)
        targets = buf[1:].view(B, T)
        
        # update position
        self.current_pos += B * T * self.ddp_world_size
        # reset position if next batch exceeds the length of tokens
        if self.current_pos + B * T + 1 >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_pos = self.rank * self.batch_size * self.seq_len
        
        return inputs, targets
    
    def load_tokens(self, path: str):
        tokens = np.load(path)
        tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens
    
    def reset(self):
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_pos = self.rank * self.batch_size * self.seq_len
    
# cos-decay with warm-up learning rate scheduler
class LearningRateScheduler:
    def __init__(self, lr: float, warm_up_steps: int, max_steps: int, **kwargs):
        self.lr = lr
        self.warm_up_steps = warm_up_steps
        self.max_steps = max_steps
        self.lowert_pct = 0.1
        self.min_lr = self.lowert_pct * self.lr

    def __call__(self, step: int):
        """
        step: int
            The training step, starting from 0
        """
        if step < self.warm_up_steps:
            return (step + 1) / (self.warm_up_steps) * self.lr
        elif step > self.max_steps:
            return self.min_lr
        # cos-decay 
        else:
            coeff = 1.0 + math.cos(math.pi * (step - self.warm_up_steps) / (self.max_steps - self.warm_up_steps))
            return self.min_lr + 0.5 * coeff * (self.lr - self.min_lr)


def valid_context(model, valid_loader, step, master_process, rank, world_size, eval_per_steps=50, eval_round: int=1):
    # validation loop
    if step == 0 or (step + 1) % eval_per_steps == 0:
        print("="*25 + f" BEGIN EVALUATION ROUND={eval_round} " + "="*25) if master_process else None
        # eval on pre-training data
        model.eval()
        valid_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                inputs, targets = valid_loader.next_batch()
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, loss = model(inputs, targets)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG) if ddp else None

        # eval on helloswag
        acc_norm = evaluate_ddp(model, rank, world_size, split='val', debug=False)

        if master_process:
            msg = f" step {step:4d}, valid loss: {val_loss_accum.item():.4f}, acc on helloswag: {acc_norm:.4f} "
            print("="*8 + msg + "="*8)
        eval_round += 1
        
    return eval_round

def train_context(model, train_loader, step, optimizer, lr_scheduler, master_process):
    # training loop
    model.train()
    t0 = time.time() 
    loss_accum = 0.0
    optimizer.zero_grad()
    for micro_step in range(GRAD_ACCUM_STEPS):
        inputs, targets = train_loader.next_batch()
        inputs, targets = inputs.to(device), targets.to(device)
        # cancel sync utill the last micro-step
        model.require_backward_grad_sync = micro_step == GRAD_ACCUM_STEPS - 1 \
            if ddp else model.require_backward_grad_sync
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(inputs, targets)
        loss = loss / GRAD_ACCUM_STEPS
        loss_accum += loss.detach()
        loss.backward()
    # sync loss_accum
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) if ddp else None
    
    # gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # set learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_scheduler(step)

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    time_used = (t1 - t0)
    tok_seen = train_loader.batch_size * train_loader.seq_len * GRAD_ACCUM_STEPS * ddp_world_size
    tok_per_sec = tok_seen / time_used
    if master_process: 
        print(f"step {step:4d}, loss {loss_accum.item():.4f}, norm {norm:.4f}, time {time_used*1000:.2f}ms, tok/sec {tok_per_sec:.2f}")


if __name__ == '__main__':
    # set DDP
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        # use DDP training
        assert torch.cuda.is_available()
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        # device = f'cuda:{ddp_local_rank}'
        device = 'cuda:0'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # master do logging
    else:
        # use single GPU training
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = 'cuda'
        master_process = True

    # set random seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # set percision
    torch.set_float32_matmul_precision('high')

    # config batch_size
    batch_size = 2**19 # ~ 0.5M tokens 
    micro_batch_size = 8
    seq_len = 1024
    assert batch_size % (micro_batch_size * seq_len * ddp_world_size) == 0
    GRAD_ACCUM_STEPS = batch_size // (micro_batch_size * seq_len * ddp_world_size)
    # create data loader
    train_loader = DataLoaeder(batch_size=micro_batch_size, seq_len=seq_len,
                               ddp_world_size=ddp_world_size, rank=ddp_rank, split='train')
    valid_loader = DataLoaeder(batch_size=micro_batch_size, seq_len=seq_len,
                               ddp_world_size=ddp_world_size, rank=ddp_rank, split='valid')

    if master_process:
        print("Total desired batch size: ", batch_size)
        print("=> accumulate gradient steps: ", GRAD_ACCUM_STEPS)

    # create model
    model = GPT(GPT2Config(vocab_size=50304))
    model.to(device)
    # compile model
    # model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank]) 
        raw_model = model.module
    else:
        raw_model = model

    # create optimizer and lr_scheduler
    optimizer = raw_model.build_optimizer(lr=6e-4, weight_decay=0.1)
    lr_scheduler = LearningRateScheduler(lr=6e-4, warm_up_steps=10, max_steps=100)

    print("="*30 + " BEGIN TRAININ " + "="*30) if master_process else None
    num_steps = 50
    eval_round = 1
    for step in range(num_steps):
        # validation loop
        eval_round = valid_context(model, valid_loader, step, master_process, ddp_rank, ddp_world_size, eval_per_steps=25, eval_round=eval_round)
        # training loop
        train_context(model, train_loader, step, optimizer, lr_scheduler, master_process)
    
    # destroy process group
    destroy_process_group() if ddp else None