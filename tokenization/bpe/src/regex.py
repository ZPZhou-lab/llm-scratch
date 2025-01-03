from typing import List, Tuple, Dict, Any, Optional, Union
import regex as re
from .basic import BasicTokenizer

GPT2PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?\p{P}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(BasicTokenizer):
    def __init__(self, pattern=None, **kwargs):
        super().__init__(**kwargs)
        self.pattern = GPT4PAT if pattern is None else pattern

    def _build_bytes_tokens(self, texts: str | List[str]) -> List[List[int]]:
        """
        build a list of byte tokens from a list of texts

        Args:
            texts: list of texts
        """
        if isinstance(texts, str):
            texts = [texts]

        tokens = []
        pattern = re.compile(self.pattern)
        for text in texts:
            # forced split using the pattern
            splits = re.findall(pattern, text)
            # create utf-8 byte tokens
            tokens.extend(
                [list(map(int, part.encode("utf-8"))) for part in splits]
            )
        return tokens