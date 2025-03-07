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
