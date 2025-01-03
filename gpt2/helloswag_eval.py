"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""

import os
import json
import tiktoken
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn import functional as F

DATA_CACHE_DIR = "../dataset/hellaswag/data"
# get encoder
enc = tiktoken.get_encoding('gpt2')

def render_example(example: dict):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - `tokens` (the tokens of context + completion), with shape (4, max_len) for 4 completions
    - `mask` (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - `label` (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx     = example["ctx"]
    label   = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": enc.encode(ctx),
        "ending_tokens": [],
    }

    # gather up all the tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end) # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(data['ctx_tokens'] + end_tokens)
        mask_rows.append([0]*len(data['ctx_tokens']) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples(split: str):
    # there are 10,042 examples in total in val
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


def calculate_completion_prediction(
    logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor, label: int      
):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)

    # get loss (loglikelihood) with shape (4 * max_len - 1) -> (4, max_len - 1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask

    # sum and divide by the number of 1s in the mask, (4, max_len - 1) -> (4,)
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)

    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred = sum_loss.argmin().item()
    pred_norm = avg_loss.argmin().item()

    return pred, pred_norm, avg_loss

@torch.no_grad()
def evaluate(model, split: str="val", debug: bool=False):
    torch.set_float32_matmul_precision('high') # use tf32
    device = next(model.parameters()).device

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples(split=split):
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(tokens) # (4, max_len, vocab_size)
        # calculate the completion prediction
        pred, pred_norm, avg_loss = calculate_completion_prediction(logits, tokens, mask, label)

        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        acc_norm = num_correct_norm / num_total

        # debug: pretty print a few examples, and the losses in each case
        if num_total < 10 and debug:
            print(f"=> Example {num_total}")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")
        
    acc_norm = num_correct_norm / num_total
    return acc_norm
    
@torch.no_grad()
def evaluate_ddp(model, rank: int, ddp_world_size: int, split: str="val", debug: bool=False):
    """
    evaluate in DDP mode
    """
    torch.set_float32_matmul_precision('high')
    device = next(model.parameters()).device

    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples(split=split)):
        if i % ddp_world_size != rank:
            continue
        # get the example
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(tokens) # (4, max_len, vocab_size)
        # calculate the completion prediction
        pred, pred_norm, avg_loss = calculate_completion_prediction(logits, tokens, mask, label)

        # accumulate stats
        num_total += 1
        num_correct_norm += int(pred_norm == label)

    # all-reduce
    num_total = torch.tensor(num_total, device=device)
    num_correct_norm = torch.tensor(num_correct_norm, device=device)
    dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
    # calculate accuracy
    acc_norm = num_correct_norm.item() / num_total.item()

    return acc_norm