import string
from typing import Literal
import torch
from torch import nn
from einops import rearrange

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

        self.register_buffer("beta_t_tensors", beta_t)
        self.register_buffer("one_minus_beta_t_cum_tensors", self._init_one_minut_beta_cum())

        self.register_buffer("qtcum_tensors", self._init_qtcum())
        self.register_buffer("qt_tensors", self._init_qt())

    def apply_qtcum(self, x: torch.Tensor, t: int=1) -> torch.Tensor:
        return Forward.batch_sample_multinomial(self.qtcum_tensors[t][x])
    
    @staticmethod
    def batch_sample_multinomial(dists: torch.Tensor) -> torch.Tensor:
        samples = torch.multinomial(rearrange(dists, "B S K -> (B S) K"), 1)
        return rearrange(samples, "(B S) 1 -> B S", B=dists.shape[0])

    def _init_one_minut_beta_cum(self) -> torch.Tensor:
        return torch.cumprod(1 - self.beta_t_tensors, dim=0)[:, None, None]
    
    def _init_qtcum(self) -> torch.Tensor:
        device = self.one_minus_beta_t_cum_tensors.device

        if self.noise_type == "uniform":
            qtcum = self.one_minus_beta_t_cum_tensors * torch.eye(self.K, device=device) + \
                    (1-self.one_minus_beta_t_cum_tensors) * torch.ones(self.K, self.K, device=device) / self.K
        elif self.noise_type == "absorbing":
            qtcum = self.one_minus_beta_t_cum_tensors * torch.eye(self.K, device=device) + \
                    (1-self.one_minus_beta_t_cum_tensors) * torch.ones(self.K, device=device)[:, None] @ \
                    torch.nn.functional.one_hot(torch.tensor([self.absorbing_id], device=device), num_classes=self.K).float()
        
        qtcum[:, :, self.pad_id] = 0
        qtcum[:, self.pad_id, :] = 0
        qtcum[:, self.pad_id, self.pad_id] = 1
        qtcum = qtcum / qtcum.sum(dim=-1, keepdim=True)
        return qtcum

    def _init_qt(self) -> torch.Tensor:
        device = self.beta_t_tensors.device
        beta_expanded = self.beta_t_tensors[:, None, None]
        
        if self.noise_type == "uniform":
            qt = (1-beta_expanded) * torch.eye(self.K, device=device) + \
                 beta_expanded * torch.ones(self.K, self.K, device=device) / self.K
        elif self.noise_type == "absorbing":
            qt = (1-beta_expanded) * torch.eye(self.K, device=device) + \
                 beta_expanded * torch.ones(self.K, device=device)[:, None] @ \
                 torch.nn.functional.one_hot(torch.tensor([self.absorbing_id], device=device), num_classes=self.K).float()
        
        qt[:, :, self.pad_id] = 0
        qt[:, self.pad_id, :] = 0
        qt[:, self.pad_id, self.pad_id] = 1
        qt = qt / qt.sum(dim=-1, keepdim=True)
        return qt

    def qt(self, x: torch.Tensor, t: int) -> torch.Tensor:
            device = self.beta_t_tensors.device

            if self.noise_type == "uniform":
                qt = (1 - self.beta_t_tensors[t]) * torch.eye(self.K, device=device) + \
                    self.beta_t_tensors[t] * torch.ones(self.K, self.K, device=device) / self.K
                    
            elif self.noise_type == "absorbing":
                absorb_idx = torch.tensor([self.absorbing_id], device=device)
                
                absorb_mat = torch.ones(self.K, device=device)[:, None] @ \
                            torch.nn.functional.one_hot(absorb_idx, num_classes=self.K).to(torch.float)

                qt = (1 - self.beta_t_tensors[t]) * torch.eye(self.K, device=device) + \
                    self.beta_t_tensors[t] * absorb_mat

            qt[:, self.pad_id] = 0
            qt[self.pad_id, :] = 0
            qt[self.pad_id, self.pad_id] = 1
            qt = qt / qt.sum(dim=-1, keepdim=True)
            
            return self.batch_sample_multinomial(qt[x])

class Tokenizer:
    def __init__(self, vocab: list=string.printable, max_len=10):
        self.id_to_tok = {i: tok for i, tok in enumerate(vocab)}
        self.dict = vocab
        self.size = len(vocab)
        self.max_len = max_len

        self.pad_token = "<PAD>"
        self.pad_id = self.size + 1
        self.id_to_tok[self.pad_id] = self.pad_token

        self.mask_token = "<MASK>"
        self.mask_id = self.size
        self.id_to_tok[self.mask_id] = self.mask_token

        self.tok_to_id = {tok: i for i, tok in self.id_to_tok.items()}

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
forward = Forward(beta_t=0.03, K=len(string.printable)+2, T=50, noise_type="absorbing", absorbing_id=len(string.printable), pad_id=len(string.printable)+1)
tokenizer = Tokenizer(string.printable)

x = tokenizer.encode(["hello", "bye"])
print("Encoded:", x)
print("Decoded:", tokenizer.decode(x))

xhat = torch.clone(x)
for i in range(50):
    xhat = forward.qt(x=xhat, t=i)
    print(tokenizer.decode(xhat))
