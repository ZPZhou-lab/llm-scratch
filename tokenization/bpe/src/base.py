from typing import List, Tuple, Dict, Any, Optional, Union
from tqdm import tqdm
from abc import ABC, abstractmethod
import unicodedata


def get_byte_pair_counts(
    byte_list: list, 
    pair_counts: Optional[Dict[Tuple[int, int], int]] = None
) -> Dict[Tuple[int, int], int]:
    """
    count the frequency of each pair of bytes in-place

    Args:
        byte_list: list of bytes
        pair_counts: dictionary of pair counts
    """
    counts = {} if pair_counts is None else pair_counts
    if len(byte_list) < 2:
        return counts
    
    for pair in zip(byte_list, byte_list[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    # sort by counts
    return counts


def merge_byte_pair(byte_list: list, pair: tuple, token_id: int) -> list:
    """
    merge the pair of bytes into a single byte token using the given token_id

    Args:
        byte_list: list of bytes
        pair: pair of bytes to merge
        token_id: token id to use for the merged pair
    """
    merged_bytes = []
    i = 0
    while i < len(byte_list):
        if i < len(byte_list) - 1 and (byte_list[i], byte_list[i+1]) == pair:
            merged_bytes.append(token_id)
            i += 2
        else:
            merged_bytes.append(byte_list[i])
            i += 1
    return merged_bytes


class BaseTokenizer:
    def __init__(self, **kwargs):
        self.vocab      = None # map token_id into byte token
        self._merge_map = None # map byte pair into a new token id
        self.pattern    = None # regex pattern for tokenization
        self.special_tokens = kwargs.get('special_tokens', {})
        self._inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

    @abstractmethod
    def train(self, 
        texts: Union[str, List[str]],
        vocab_size: int,
        verbose: bool = True
    ) -> None:
        raise NotImplementedError("train method must be implemented")

    @property
    def vocab_size(self) -> int:
        self._is_trained()
        return len(self.vocab)

    @property
    def merges_map(self) -> Dict[Tuple[int, int], int]:
        self._is_trained()
        merges_map = []
        for pair, token_id in self._merge_map.items():
            pair_a = self.vocab[pair[0]].decode('utf-8', errors='replace')
            pair_b = self.vocab[pair[1]].decode('utf-8', errors='replace')
            merged = self.vocab[token_id].decode('utf-8', errors='replace')
            merge_info = f"[{pair_a}][{pair_b}] -> [{merged}]"
            merges_map.append(merge_info)
        
        return merges_map
    
    def add_special_tokens(self, special_tokens: Dict[str, str]) -> None:
        """
        add special tokens to the tokenizer

        Args:
            special_tokens: dictionary of special tokens
        """
        self.special_tokens.update(special_tokens)
        self._inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
    
    def encode(self, text: str) -> List[int]:
        """
        encode a text into a list of token ids

        Args:
            text: input text
        """
        self._is_trained()
        tokens = list(map(int, text.encode('utf-8')))

        # iteratively merge tokens
        while len(tokens) > 1:
            pair_counts = get_byte_pair_counts(tokens)
            # get the pair with the minimum merged token_id
            pair = min(pair_counts, key=lambda x: self._merge_map.get(x, float('inf')))
            if pair not in self._merge_map:
                break
            # merge the pair
            token_id = self._merge_map[pair]
            tokens = merge_byte_pair(tokens, pair, token_id)
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """
        decode a list of token ids into a text

        Args:
            tokens: list of token ids
        """
        self._is_trained()
        return b"".join([self.vocab[token_id] for token_id in tokens])\
            .decode('utf-8', errors='replace')

    def _is_trained(self):
        if self.vocab is None:
            raise ValueError("Tokenizer has not been trained yet")
    
    def save(self, path: str) -> None:
        """
        save the tokenizer to a file
        .model: tokenizer metadata used for encoding and decoding, can be loaded using load()
        .vocab: a pretty printed version for human inspection only

        Args:
            path: path to save the tokenizer
        """
        # create .model file
        with open(path + '.model', 'w') as f:
            # write the version, pattern and merges
            f.write(f"version: v1\n")
            f.write(f"pattern:\n")
            f.write(f"{self.pattern}\n")
            
            # write the special tokens
            f.write(f"num_special_tokens: {len(self.special_tokens)}\n")
            for token, token_id in self.special_tokens.items():
                f.write(f"{token} {token_id}\n")
            # write the merges
            for pair, token_id in self._merge_map.items():
                f.write(f"{pair}->{token_id}\n")

        # create .vocab file
        inv_merges_map = {idx: pair for pair, idx in self._merge_map.items()}
        with open(path + '.vocab', 'w') as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                # find the children of this token, if any
                if idx in inv_merges_map:
                    # if this token has children, render it nicely as a merge
                    pair = inv_merges_map[idx]
                    s0 = render_token(self.vocab[pair[0]])
                    s1 = render_token(self.vocab[pair[1]])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")
        
    def load(self, path: str) -> None:
        """
        load the tokenizer from a file
        """

        assert path.endswith('.model'), "path must end with .model"
        # init the tokenizer
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        merge_map, self.special_tokens = {}, {}

        # load the .model file
        with open(path, 'r') as f:
            # read the version, pattern and merges
            version = f.readline().strip().split()[-1]

            # read the pattern
            f.readline() # skip the pattern line
            self.pattern = f.readline().strip()
            self.pattern = None if self.pattern == 'None' else self.pattern
            
            # read the special tokens
            num_special_tokens = int(f.readline().strip().split()[-1])
            for _ in range(num_special_tokens):
                token, token_id = f.readline().strip().split()
                self.special_tokens[token] = int(token_id)
            # read the merges
            for line in f:
                pair, token_id = line.strip().split('->')
                pair = tuple(map(int, pair[1:-1].split(', ')))
                merge_map[pair] = int(token_id)
                self.vocab[int(token_id)] = self.vocab[pair[0]] + self.vocab[pair[1]]
        
        # add into attributes
        self._merge_map = merge_map
        self._inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}


def render_token(t: bytes) -> str:
    def replace_control_characters(s: str) -> str:
        # we don't want to print control characters
        # which distort the output (e.g. \n or much worse)
        # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        # http://www.unicode.org/reports/tr44/#GC_Values_Table
        chars = []
        for ch in s:
            if unicodedata.category(ch)[0] != "C":
                chars.append(ch) # this character is ok
            else:
                chars.append(f"\\u{ord(ch):04x}") # escape
        return "".join(chars)
    
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s