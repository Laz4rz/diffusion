## D3PM

Main idea is that one can corrupt the data by iteratively applying kernels. Kernels are usually uniform or absorbing states. For example uniform transition matrix is:

$$ Q^{uniform}=(1-\beta_t)\bold{I}+\beta_t\mathbf{1}\mathbf{1}^T/K $$

$$[Q_t]_{ij} = 
\begin{cases}
1 - \dfrac{K-1}{K}\,\beta_t & \text{if } i = j,\\[4pt]
\dfrac{1}{K}\,\beta_t       & \text{if } i \ne j ,
\end{cases}$$

Or absorbing state transition:

$$Q^{absorb}=(1-\beta_t)\bold{I}+\beta_t\mathbf{1}e^T_m$$

$$[Q_t]_{ij} =
\begin{cases}
1-\beta_t & \text{if } i = j \neq m,\\[4pt]
1         & \text{if } i = j = m,\\[4pt]
\beta_t   & \text{if } j = m,\; i \neq m .
\end{cases}$$

where $e^T_M$ is a one-hot vector of the MASK token. These kernels have some properties they have to obey (look paper) and allow us to sample for the forward process $q(x_t|x_0)$.

The forward process has noise scheduling, linear for uniform kernel and linearly interpolating mutual information for absorbing-state kernel ($(T-t+1)^{-1}$, appendix). 

Then we have to do the reverse process and its done using the forward process posterior $q(x_{t-1}|x_t, x_0)$. The posterior can be expanded using Bayes so we have:

$$q(x_{t-1}|x_t,x_0)=\frac{q(x_t|x_{t-1}, x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)}=\\=\mathrm{Cat}(x_{t-1};p=\frac{\mathbf{x}_t\mathbf Q_t^T \odot \mathbf x_0 \mathbf{\bar Q}_{t-1}}{\mathbf x_0 \mathbf{\bar Q}_{t-1}\mathbf x_t}), \;\mathrm{ with} \; \mathbf{\bar Q}_{t}=\mathbf Q_1\mathbf Q_2...\mathbf Q_t$$

Where $q(x_{t}|x_{t-1},x_0)=q(x_{t}|x_{t-1})$ due to Markov property. This true reverse distribution is perfect for training, but at inference we do not have any $x_0$ so we need something like $p_\theta(x_{t-1}|x_t)$ that will do the denoising step purely based on $x_t$ and $t$.

This is not implicit in the paper, but the $p_\theta$ network is more like $p_\theta(x_{t-1}|x_t;t)$, as both $x_t$ and $t$ are it's inputs. Typical $p_\theta$ has the main network + time embedding mechanism. The time embedding can be for example some version of positional embedding (like Sinusoidal).

Funnily, D3PM follows another paper and instead of predicting $p_\theta$ directly it learns to predict $\tilde p_\theta(x_0|x_t)$. So it directly aims at going towards the final prediction, without trying to learn all the tiny denoising jumps. Then tiny jump is calculated as:

$$p_\theta(x_{t-1}|x_t)\propto\sum_{\tilde x_0}q(x_{t-1}|x_t,\tilde x_0)\cdot\tilde p_\theta(\tilde x_0|x_t)$$

Full Bayes form is belowe, but as we see it's normalized by default.

$$p_\theta(x_{t-1}\mid x_t)
\;=\;
\frac{\displaystyle
      \sum_{\tilde x_0} q(x_{t-1}\mid x_t,\tilde x_0)\;p_\theta(\tilde x_0\mid x_t)
     }{\displaystyle
      \sum_{x'_{t-1}}\sum_{\tilde x_0} q(x'_{t-1}\mid x_t,\tilde x_0)\;p_\theta(\tilde x_0\mid x_t)
     }
\;=\;
\sum_{\tilde x_0} q(x_{t-1}\mid x_t,\tilde x_0)\;p_\theta(\tilde x_0\mid x_t)$$

```python
# Compute unnormalized mixture
logits = torch.zeros(K)
for x0_candidate in range(K):
    posterior = compute_q(xt, x0_candidate)  # shape [K]
    weight = model(xt)[:, x0_candidate]      # p(x0_candidate|xt)
    logits += posterior * weight

# Normalize
probs = logits / logits.sum(dim=-1, keepdim=True)
```

So now, for consistency check: if we perfectly predict specific $x_0$ then the whole term simplifies to:

$$p_\theta(x_{t-1}|x_t)=q(x_{t-1}|x_t,\tilde x_0)$$

Knowing what we want to achieve and how, we move to the loss function: 

$$L_{vb} = E_{q(x_0)} \left[ D_{KL}[q(x_T |x_0) || p(x_T)] + \sum_{t=2}^T E_{q(x_t |x_0)} [D_{KL}[q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t)]] - E_{q(x_1|x_0)}[\log p_\theta(x_0|x_1)] \right]$$

$$L_{\text{simple}} = E_t \left[ E_{q(x_0, x_t)} [ -\log p_\theta(x_0|x_t) ] \right]$$

$$L_{\lambda} = L_{vb} + \lambda L_{simple}$$

## Math cheatsheet

$
\begin{aligned}
\textbf{Latent variable} &:\quad z \\
\textbf{Observed data} &:\quad D \\[4pt]
\textbf{Prior} &:\quad p(z) \\
\textbf{Likelihood} &:\quad p(D \mid z) \\[4pt]
\textbf{Joint} &:\quad p(z, D) \;=\; p(z)\,p(D \mid z) \;=\; p(z \mid D)\,p(D) \\[4pt]
\textbf{Evidence / marginal likelihood} &:\quad p(D) \;=\; \int p(z)\,p(D \mid z)\,dz \\
\textbf{Posterior} &:\quad p(z \mid D) \;=\; \dfrac{p(z)\,p(D \mid z)}{p(D)} \\
\textbf{Surrogate} &:\quad q(z) \\
\end{aligned}
$

### KL Divergence
$$
D_{\mathrm{KL}}\!\left(q(z)\,\|\,p(z\mid D)\right)
= \mathbb{E}_{z\sim q(z)}\!\left[ \log \frac{q(z)}{p(z\mid D)} \right]
= \int\!\cdots\!\int q(z)\,\log\!\frac{q(z)}{p(z\mid D)}\,dz_0\cdots dz_{d-1}.
$$

### KL Divergence between surrogate and posterior (ELBO derivation)
$$
D_{KL}(q(z)||p(z|D)) = \mathbb E_{z\sim q(z)}[\log\frac{q(z)\cdot p(D)}{p(z,D)}]=\int_{\bar{z}}q(z)\log\frac{q(z)\cdot p(D)}{p(z,D)}d\bar{z}\\$$

We use the join identity, split and move back to expectations.

$$=\int_{\bar{z}}q(z)\log\frac{q(z)}
{p(z,D)}d\bar{z}+\int_{\bar{z}}q(z)\log p(D)d\bar{z}=\mathbb E_{z\sim q(z)}[\log\frac{q(z)}{p(z,D)}] + \mathbb E_{z\sim q(z)}[\log p(z,D)]
$$

Now the second term is nicely constant. 

$$
=\mathbb E_{z\sim q(z)}[\log\frac{q(z)}{p(z,D)}] + \log p(z,D) = \underbrace{-\mathbb E_{z\sim q(z)}[\log\frac{p(z,D)}{q(z)}]}_{\mathcal{L}(q)} + \log p(z,D) 
$$

So now we have the KL divergence as one term dependent on the surrogate $q$ and one constant (evidence, negative) term:

$$
\mathrm{KL}
\;=\;
-\,\mathcal{L}(q)
\;+\;
\underbrace{\log \overbrace{p(D)}^{\text{marginal (constant)}}}_{\text{evidence}}
$$

We know that KL is a distance so $\mathrm{KL} \ge 0 \rightarrow $

### Important:
https://yunfanj.com/blog/2021/01/11/ELBO.html \
https://www.wpeebles.com/DiT