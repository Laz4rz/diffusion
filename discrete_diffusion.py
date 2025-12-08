import string
from typing import Literal
import torch
from torch import nn
from torch.nn import functional as F


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
        if isinstance(t, int):
            return Forward.batch_sample_multinomial(self.qtcum_tensors[t][x])
        else:
            return Forward.batch_sample_multinomial(self.qtcum_tensors[torch.randint(0, self.steps, size=(x.shape[0],))][:, x])
    
    @staticmethod
    def batch_sample_multinomial(dists: torch.Tensor) -> torch.Tensor:
        samples = torch.multinomial(dists.view(-1, dists.shape[-1]), 1)
        return samples.view(dists.shape[:-1])

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
# forward = Forward(beta_t=0.03, K=len(string.printable)+2, T=50, noise_type="absorbing", absorbing_id=len(string.printable), pad_id=len(string.printable)+1)
# tokenizer = Tokenizer(string.printable)

# x = tokenizer.encode(["hello", "bye"])
# print("Encoded:", x)
# print("Decoded:", tokenizer.decode(x))


# xhat = torch.clone(x)
# for i in range(50):
#     xhat = forward.qt(x=xhat, t=i)
#     print(tokenizer.decode(xhat))

# xt = forward.apply_qtcum(x0, t)
# x0pred = model(xt)
# 
# Lvb := Dkl[q(xt-1 | xt, x0) || p(xt-1 | xt)]
# p(xt-1 | xt) = sum_x'0 (q(xt, xt-1 | x'0) * p(x'0 | xt)) / q(xt | x'0)

# q: [T, K, K]
#
# q(xt, xt-1 | x'0) = q(xt | xt-1, x'0) * q(xt-1 | x'0) =(Markov)= q(xt | xt-1) * q(xt-1 | x'0)
# when calculating loss:
# q(xt | xt-1) = qt[t, :, xt_id] [K]
# (explicit sum) q(xt-1 | x'0) = qtcum[t-1, x'0_id, :] [K]
# (implicit sum) q(xt-1 | x'0) = qtcum[t-1, :, :] [K, K]
#
# q(xt-1|xt, x0) = [q(xt | xt-1) * q(xt-1 | x0)] / q(xt | x0)
# q(xt-1|xt, x0) = qt[t, :, xt_id] * qtcum[t-1, x0_id, :] / qtcum[t, x0_id, xt_id]

# sample x0 -> x0
# sample t -> t
# apply t to x0 -> xt
# pass xt to model -> pred

def hybrid_loss(t, x0, xt, pred, pad_id, forward):
    # direct loss
    lax = F.cross_entropy(pred.flatten(0, 1), x0.flatten(0, 1), ignore_index=pad_id)

    B, L, K = pred.shape

    # NVLB (trajectory, t to t-1) loss
    qt = forward.qt_tensors
    qtcum = forward.qtcum_tensors
    probs = F.softmax(pred, dim=-1)
    # weight x0: x_t-1 by corresponding model prediction of x0
    # "based on model prediction, onto which t-1 token distribution I land with what probability"
    qp_sum = probs @ qtcum[t-1]
    # probs: [1, K] (Row vector)
    # qtcum: [K, K] (Matrix)
    # weigh tokens by the probability of landing on correct xt
    constraint = qt[t, :, xt.reshape(-1)].T.reshape(B, L, K)
    p_prev = qp_sum * constraint
    p_prev /= (p_prev.sum(dim=2, keepdim=True) + 1e-8)

    q_prev_cond = qt[t, :, xt.reshape(-1)].T.reshape(B, L, K) * qtcum[t-1, x0, :]
    q_prev_cond /= (q_prev_cond.sum(dim=2, keepdim=True) + 1e-8)

    pad_mask = (x0 != pad_id)
    p_prev = p_prev.clamp_min(1e-8)[pad_mask]
    q_prev_cond = q_prev_cond.clamp_min(1e-8)[pad_mask]
    lvb = F.kl_div(input=p_prev.log(), target=q_prev_cond, reduction="batchmean")

    return lax+lvb

def train_batch(model, forward, optimizer, x0):
    model.train()
    optimizer.zero_grad()

    # batch_size = x0.shape[0]

    # t = torch.randint(low=0, high=forward.steps, size=batch_size, device=x0.device)
    t = torch.randint(
            low=1,
            high=forward.steps,
            size=(1,),
            device=x0.device,
        ).item()    
    xt = forward.apply_qtcum(x0, t)
    pred = model(xt)

    loss = hybrid_loss(t, x0, xt, pred, forward.pad_id, forward)

    loss.backward()
    optimizer.step()

class SimpleModel(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)      # [B, L, d_model]
        logits = self.linear(emb)    # [B, L, vocab_size]
        return logits


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer(string.printable, max_len=8)
    K = tokenizer.pad_id + 1
    T = 50

    forward = Forward(
        beta_t=0.03,
        K=K,
        T=T,
        pad_id=tokenizer.pad_id,
        noise_type="absorbing",
        absorbing_id=tokenizer.mask_id,
    ).to(device)

    model = SimpleModel(vocab_size=K, d_model=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    words = ["hello", "world", "cat", "dog", "trick", "noise", "model", "data"]
    x0 = tokenizer.encode(words, device=device, padding_strategy="max_len")  # [8, L]

    for step in range(2000):
        loss = train_batch(model, forward, optimizer, x0)
        if step % 200 == 0:
            print(f"step {step:4d} | loss {loss:.4f}")

    model.eval()
    with torch.no_grad():
        t_eval = T - 1
        xt = forward.apply_qtcum(x0, t_eval)
        pred = model(xt)
        x0_hat = pred.argmax(dim=-1)

        print("target:", tokenizer.decode(x0))
        print("recon: ", tokenizer.decode(x0_hat))


if __name__ == "__main__":
    main()
