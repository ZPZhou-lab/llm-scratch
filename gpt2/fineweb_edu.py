# fineweb-edu dataset
import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = 'fineweb_edu_10B'
total_size = int(1e10)  # 10B tokens
shard_size = int(1e8)   # 100M tokens per shard
num_shards = total_size // shard_size

# init tokenizers
enc = tiktoken.get_encoding('gpt2')
VOCAB_SIZE = 2**16 # less than 50257
EOT = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
    tokens = [EOT] # the EOT delimits all documents
    tokens.extend(enc.encode_ordinary(doc['text']))
    tokens_np = np.array(tokens)
    assert (tokens_np >= 0).all() and (tokens_np < VOCAB_SIZE).all()
    tokens_np = tokens_np.astype(np.uint16)
    return tokens_np

def write_shard(file: str, shard: np.ndarray):
    np.save(file, shard)

if __name__ == "__main__":
    # create the cache the local directory
    DATA_CACHE_DIR = os.path.join(
        os.path.dirname(__file__), local_dir
    )
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # load the dataset
    fineweb = load_dataset("/home/xavierzhou/project/llm_scratch/fineweb-edu/sample/10BT/", split='train')

    # tokenize all documents using mp
    nproc = max(1, os.cpu_count() // 2)
    with mp.Pool(nproc) as pool:
        # pre-allocation buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        shard_index = 0
        pbar = None

        # iter on pool
        for tokens in pool.imap(tokenize, fineweb, chunksize=16):
            if token_count + len(tokens) < shard_size:
                # append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if pbar is None:
                    pbar = tqdm(total=shard_size, unit='token', desc=f"Shard {shard_index}")
                pbar.update(len(tokens))
            else:
                split = "valid" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f'fineweb_edu_{split}_{shard_index:06d}.np')
                # split the document
                remainder = shard_size - token_count
                pbar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                # write shard to disk
                write_shard(filename, all_tokens_np)
                shard_index += 1
                pbar.close()
                pbar = None
                # populate next shard
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder
        
        # write any remaining tokens to last shard
        if token_count != 0:
            split = "valid" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f'fineweb_edu_{split}_{shard_index:06d}.np')
            write_shard(filename, all_tokens_np[:token_count])