import string
import torch
import numpy as np

K = string.printable
T = 10

print("K categories:", len(K))
print("T time steps:", T)

def qt_uniform(beta_t, K, length: int = 5) -> torch.Tensor:
    return (1-beta_t) * torch.eye(length) + beta_t / K * torch.ones(length, length)

class forward:
    def __init__(self, beta_t: torch.Tensor | float, K: int, seq_len: int, T: int):
        if isinstance(beta_t, float):
            beta_t = torch.tensor([beta_t]).expand(T)
        else:
            assert len(beta_t) == T
            beta_t = beta_t
        
        self.seq_len = seq_len
        self.steps = T

        self.register_buffer("beta_t", beta_t)
        self.register_buffer("one_minus_beta_t_cum", self._init_one_minut_beta_cum())
        self.register_buffer("qtcum_uniform", self._init_qtcum_uniform())

    def _init_one_minut_beta_cum(self) -> torch.Tensor:
        log_1_minus_beta = np.log(1 - self.beta_t)
        log_cumprobs = np.cumsum(log_1_minus_beta)
        return torch.tensor(np.exp(log_cumprobs), dtype=torch.float32)[:, None,] # would inherit float64 from numpy
    
    def _init_qtcum_uniform(self) -> torch.Tensor:
        return self.one_minus_beta_t_cum * torch.eye(K) + (1-self.one_minus_beta_t_cum) * torch.ones(K, K) / K
    
    def qt_uniform(self, t):
        (1-self.beta_t[t]) * torch.eye(self.seq_len) + self.beta_t[t] * torch.ones(self.seq_len, self.seq_len)


class Tokenizer:
    def __init__(self, dictionary: list):
        self.id_to_tok = {i: tok for i, tok in enumerate(dictionary)}
        self.tok_to_id = {tok: i for i, tok in self.id_to_tok.items()}
        self.dict = dictionary
        self.size = len(dictionary)

    def encode(self, x: list[str]) -> torch.Tensor:
        return torch.tensor([self.tok_to_id[tok] for tok in x])       
    
    def decode(self, x: torch.Tensor) -> str:
        return "".join([self.id_to_tok[i.item()] for i in x])

tokenizer = Tokenizer(string.printable)
