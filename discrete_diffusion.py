import string
from typing import Literal
import torch
from torch import nn
import numpy as np


class Forward(nn.Module):
    def __init__(
            self, 
            beta_t: torch.Tensor | float, 
            K: int, 
            T: int, 
            pad_id: int,
            noise_type: Literal["uniform", "absorbing"] = "uniform",
            absorbing_id: int = None
        ):
        assert noise_type == "uniform" or (noise_type == "absorbing" and absorbing_id is not None)
        super().__init__()
        if isinstance(beta_t, float):
            beta_t = torch.tensor([beta_t]).expand(T)
        else:
            assert len(beta_t) == T
            beta_t = beta_t
        
        self.steps = T
        self.K = K
        self.noise_type = noise_type
        self.absorbing_id = absorbing_id
        self.pad_id = pad_id

        self.register_buffer("beta_t", beta_t)
        self.register_buffer("one_minus_beta_t_cum", self._init_one_minut_beta_cum())

        self.register_buffer("qtcum", self._init_qtcum())

        self.qtcum[:, :, pad_id] = 0
        self.qtcum[:, pad_id, :] = 0
        self.qtcum[:, pad_id, pad_id] = 1
        self.qtcum = torch.softmax(self.qtcum, axis=1)

    def apply(self, x: torch.Tensor, t: int=1) -> torch.Tensor:
        # x: [B, S]
        probs = self.qtcum[t, x[0]]
        return torch.multinomial(probs, 1).T

    def _init_one_minut_beta_cum(self) -> torch.Tensor:
        return torch.cumprod(1 - self.beta_t, dim=0)[:, None, None]
    
    # Uniform noise
    def _init_qtcum(self) -> torch.Tensor:
        if self.noise_type == "uniform":
            return self.one_minus_beta_t_cum * torch.eye(self.K) + (1-self.one_minus_beta_t_cum) * torch.ones(self.K, self.K) / self.K
        elif self.noise_type == "absorbing":
            return self.one_minus_beta_t_cum * torch.eye(self.K) + (1-self.one_minus_beta_t_cum) * torch.ones(self.K)[:, None] @ torch.nn.functional.one_hot(torch.tensor([self.absorbing_id]), num_classes=self.K).to(torch.float)
    
    def qt_uniform(self, x, t):
        if self.noise_type == "uniform":
            qt = (1-self.beta_t[t]) * torch.eye(self.K) + self.beta_t[t] * torch.ones(self.K, self.K) / self.K
        elif self.noise_type == "absorbing":
            qt = (1-self.beta_t[t]) * torch.eye(self.K) + self.beta_t[t] * torch.ones(self.K)[:, None] @ torch.nn.functional.one_hot(torch.tensor([self.absorbing_id]), num_classes=self.K).to(torch.float)
        probs = qt[t, x[0]]
        return torch.multinomial(probs, 1).T

class Tokenizer:
    def __init__(self, vocab: list=string.printable, max_len=20):
        self.id_to_tok = {i: tok for i, tok in enumerate(vocab)}
        self.tok_to_id = {tok: i for i, tok in self.id_to_tok.items()}
        self.dict = vocab
        self.size = len(vocab)
        self.max_len = max_len

        self.pad_token = "<PAD>"
        self.pad_id = self.size + 1

        self.mask_token = "<MASK>"
        self.mask_id = self.size

    def encode(
            self, 
            texts: list[str], 
            device: torch.device = "cpu", 
            padding_strategy: Literal["max_len", "longest"] = "max_len"
        ) -> torch.Tensor:
        max_len = self.max_len if padding_strategy == "max_len" else max([len(text) for text in texts])

        batch_ids = []
        for text in texts:
            ids = [self.tok_to_id[tok] for tok in text][:max_len]
            batch_ids.append(ids + (max_len-len(ids)) * [self.pad_id])
        return torch.tensor(batch_ids, dtype=torch.long, device=device)
    
    def decode(self, x: torch.Tensor) -> str:
        decoded_batch = []
        for seq in x:
            decoded_batch.append("".join([self.id_to_tok[i.item()] for i in seq if i.item() != self.pad_id]))
        return decoded_batch

# maybe nicer if forward needed tokenizer to avoid vocab size mistakes
# I skip the padding token from K, add mask
forward = Forward(beta_t=0.1, K=len(string.printable)+2, T=50, noise_type="absorbing", absorbing_id=len(string.printable), pad_id=len(string.printable)+1)
tokenizer = Tokenizer(string.printable)

x = tokenizer.encode(["hello"])
print("Encoded:", x)
print("Decoded:", tokenizer.decode(x))

xhat = torch.clone(x)
for i in range(50):
    xhat = forward.qt_uniform(x=xhat, t=i)
    print(tokenizer.decode(xhat))
