from .base import BaseTokenizer
from .base import get_byte_pair_counts, merge_byte_pair
from typing import List, Dict, Tuple, Union
from tqdm import tqdm


class BasicTokenizer(BaseTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, 
        texts: Union[str, List[str]],
        vocab_size: int,
        verbose: bool = True
    ) -> None:
        """
        build a tokenizer from a text or a list of texts

        Args:
            texts: list of texts
            vocab_size: size of the vocabulary
            verbose: whether to display a progress bar
        """

        # number of BPE merges
        assert vocab_size >= 256, "vocab_size must be at least 256"
        num_merge_rounds = vocab_size - 256

        # create vocab and merge_map
        vocab = {idx: bytes([idx]) for idx in range(256)}
        merge_map = {}
        new_token_id = 256

        # progress bar
        pbar = tqdm(total=num_merge_rounds, ncols=120)

        # build bytes tokens
        tokens = self._build_bytes_tokens(texts)

        # iter merge rounds
        for r in range(num_merge_rounds):
            # init pair count
            pair_counts = {}
            # iter on tokens
            for tokens_ in tokens:
                pair_counts = get_byte_pair_counts(tokens_, pair_counts)
            
            # get the most frequent pair
            top_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[top_pair] == 1:
                break

            # merge the pair
            merge_map[top_pair] = new_token_id
            vocab[new_token_id] = vocab[top_pair[0]] + vocab[top_pair[1]]
            # update tokens
            tokens = [merge_byte_pair(tokens_, top_pair, new_token_id) for tokens_ in tokens]
        
            # update progress bar
            if verbose:
                pbar.set_description(f"Merge pair {top_pair} into {new_token_id}")
                pbar.update(1)

            # increment token id
            new_token_id += 1
        
        # close progress bar
        pbar.close()
        # update vocab and merge_map
        self.vocab = vocab
        self._merge_map = merge_map


    def _build_bytes_tokens(self, texts: Union[str, List[str]]) -> List[List[int]]:
        """
        build bytes tokens from a list of texts

        Args:
            text: list of texts
        """
        # create bytes tokens from texts
        if isinstance(texts, str):
            texts = [texts]
        tokens = [list(map(int, text.encode('utf-8'))) for text in texts]
        return tokens