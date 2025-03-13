import numpy as np

from abc import abstractmethod


class Tokenizer:
    bos_token:str
    bos_token_id:int
    eos_token:str
    eos_token_id:int
    special_tokens: list[str]

    @abstractmethod
    def encode(self, text:str, add_start_end=True)->list[int]:
        pass

    @abstractmethod
    def decode(self, tokens:list[int], replace=False)->str:
        pass

    @abstractmethod
    def __len__(self):
        pass
    
    def encode_batch(self, texts:list[str], add_start_end=True):
        tokens = [self.encode(txt, add_start_end) for txt in texts]
        max_len = max(len(tks) for tks in tokens)

        for tks in tokens:
            diff = max_len - len(tks)
            tks.extend([self.eos_token_id] * diff)
        
        return np.array(tokens)
