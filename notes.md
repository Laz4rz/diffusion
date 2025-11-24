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

Full Bayes form is below, but as we see it's normalized by default.

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

Prior in variational inference is a design choice of a distribution we try to "balance" or "regularize" the Surrogate with. Posterior is a data backed ground truth. ELBO can be rewritten as balancing between these two representations.

## KL Divergence
$$
D_{\mathrm{KL}}\big(q(\bar z)\;||\;p(\bar z |D)\big)
= \mathbb{E}_{\bar z\sim q(\bar z)}\left[ \log \frac{q(\bar z)}{p(\bar z\mid D)} \right]
= \int q(\bar z)\log\frac{q(\bar z)}{p(\bar z\mid D)}d\bar z \\
=\int\cdots\int q(z_0,\ldots,z_{d-1})
\log\frac{q(z_0,\ldots,z_{d-1})}{p(z_0,\ldots,z_{d-1}\mid D)}
dz_0\cdots dz_{d-1}.
$$

## KL Divergence between surrogate and posterior (ELBO derivation)

Variational Inference Surrogate Optimization (fitting some function that will model the posterior) can be written as:

$$q^\star(\underline{z})=\argmin_{q(\underline{z})\in Q}(\mathrm{KL}(q(\underline{z}) \; || \; p(\underline{z}|D))$$

The problem is that we do not and will not have the posterior, because it's exactly the thing we want to model. Hence we look closer at the KL term and do all subsequent steps. We will change the optimization problem we are working on.

$$
D_{KL}(q(\underline{z})||p(\underline{z}|D)) = \mathbb E_{\underline{z}\sim q(\underline{z})}[\log\frac{q(\underline{z})\cdot p(D)}{p(\underline{z},D)}]=\int q(\underline{z})\log\frac{q(\underline{z})\cdot p(D)}{p(\underline{z},D)}d\underline{z}\\$$

We use the Bayes rule, split and move back to expectations.

$$=\int q(\underline{z})\log\frac{q(\underline{z})}
{p(\underline{z},D)}d\underline{z}+\int q(\underline{z})\log p(D)d\underline{z}=\mathbb E_{\underline{z}\sim q(\underline{z})}[\log\frac{q(\underline{z})}{p(\underline{z},D)}] + \mathbb E_{\underline{z}\sim q(\underline{z})}[\log p(D)]
$$

Now the second term is nicely constant. 

$$
=\mathbb E_{\underline{z}\sim q(\underline{z})}[\log\frac{q(\underline{z})}{p(\underline{z},D)}] + \log p(D) = \underbrace{-\mathbb E_{\underline{z}\sim q(\underline{z})}[\log\frac{p(\underline{z},D)}{q(\underline{z})}]}_{\mathcal{L}(q)} + \log p(D) 
$$

So now we have the KL divergence as one term dependent on the surrogate $q$ and one constant (evidence, negative) term:

$$
\mathrm{KL}
\;=\;
-\,\mathcal{L}(q)
\;+\;
\underbrace{\log \overbrace{p(D)}^{\text{marginal (constant)}}}_{\text{evidence}}
$$

We know that KL is a distance so $\mathrm{KL} \ge 0 \Rightarrow \mathcal L(q) \le \log p(D)$. $\mathcal L(q)$ is so called Evidence Lower Bound or ELBO for short. 

$$\mathrm{ELBO}: \;\mathcal L(q) = \mathbb E_{\underline{z}\sim q(\underline{z})}[\log\frac{p(\underline{z}, D)}{q(\underline{z})}] \;\; \text{and} \mathcal \;\; L(q)=\log p(D) \Leftrightarrow D_{KL}(q(\underline{z})||p(\underline{z}|D)) = 0$$

Usually we will not find an ideal fit, but we will at least go towards it. Thanks to the above we can rewrite the original optimization problem to:

$$q^\star(\underline{z}) = \argmax_{q(\underline{z})\in Q} \mathcal L(q)$$

We can do that because we know that 1). $\mathcal L(q)$ can be at most equal to evidence or lower (it is it's lower bound), 2). if $\mathcal L(q)$ is equal to evidence then $\mathrm{KL}$ is minimized, which is the goal of the original problem. Hence we can rewrite:

$$\mathcal L(q) = -\mathrm{KL} + \log p(D) = \underbrace{\mathbb{E}_{\underline{z} \sim q(\underline{z})}\left[\log \frac{p(\underline{z}, D)}{q(\underline{z})}\right]}_{\text{ELBO } \mathcal{L}(q)} 
\;=\; 
\underbrace{\log p(D)}_{\text{Evidence (Constant)}} 
\;-\; 
\underbrace{D_{KL}(q(\underline{z}) \;||\; p(\underline{z}|D))}_{\text{Approximation Gap } (\ge 0)}$$

ELBO can also be rewritten as balancing between reconstruction and regularization:

$$
\mathcal{L}(q) = \underbrace{\mathbb{E}_{\underline{z} \sim q(\underline{z})}[\log p(D|\underline{z})]}_{\text{Reconstruction Error}} - \underbrace{D_{KL}(q(\underline{z}) \;||\; p(\underline{z}))}_{\text{Regularization}}
$$

## Amortized ELBO

Amortized means we use encoder neural network to learn surrogate $q_\phi(\bm z| \bm x)$.

Above math is the theoretical take on variational inference derivation of surrogate and ELBO. In practice, like when training VAEs, this would mean that we have to find surrogate for each element of the dataset (tbh I don't see why, but yeah -- ok maybe somewhat good intuition is to look at reconstruction error and realize that if $D$ was all the data we have then we would be asking how good we are at reconstructing all the data at once... See the [Gemini example intuition](./notes_misc.md#global-surrogate-in-variational-inference)).

Quick intuition is that the granularity of surrogates tell us how well we compress: 
*   **Class-based (surrogate per training class) Surrogate ($q(z|y)$):** Compresses the **Label**. It tells the decoder "Draw a Dog."
*   **Instance-based Surrogate (surrogate per training sample) ($q(z|x)$):** Compresses the **Image**. It tells the decoder "Draw a Dog with white fur, curly texture, floppy ears, facing 45 degrees left, with grass in the background."

So what can we do about it? Turns out we - of course - can train a model to approximate the surrogate for us based on the data sample $x$. This introduces the dependency on $x$.

*   **Classical VI:** Find the best parameters for $q(z)$.
*   **VAE:** Find the best weights $\phi$ for a network **Encoder$(x)$** that outputs the parameters for $q(z|x)$.

For this $x$ dependent surrogate we also assume that it is a diagonal Gaussian:

$$q_{\phi}(\bold z|\bold x)=\mathcal N(\bold z; \bold\mu_{\phi}(\bold x),\sigma^2_{\phi}(\bold x)\odot\bold I)$$

So we take data sample $\bold x$, put it in the encoder, encoder will output the $\bm\mu_{\phi}(\bold x),\bm\sigma^2_{\phi}(\bold x)$ vectors and we use them to sample $\bold z$ from Gaussian (we can't sample! but that's like 20 lines below) defined by these mean and variance vectors.

In VAEs we use the Standard Multivariate Gaussian as prior:

$$p(\bold z) = \mathcal N(z;0,\bold I)$$

The decoder part of VAE is another neural network (we will assume it is some Gaussian with fixed variance) that we will defines as:

$$
p_\theta(\mathbf{x}| \mathbf{z}) = \mathcal{N}(\mathbf{x}; \mathbf{D}_\theta(\mathbf{z}), \mathbf{I}) = \frac{1}{\sqrt{(2\pi)^k}} \exp\left(-\frac{1}{2} ||\mathbf{x} - \mathbf{D}_\theta(\mathbf{z})||^2\right)
$$

$$\log p(\bm x|\bm z) = -\frac 1 2 ||x - \mathrm D_\theta(\bm z)||^2+C$$

The log-likelihood is just copy pasted dw.

Knowing the setup we can get back to deriving ELBO for VAEs. ELBO equation for a single data point is now:

$$\mathcal L(\theta, \phi; \bold x)=\mathbb{E}_{\bold z \sim q_{\phi}(\bold z|\bold x)}[\log p_\theta(\bold x| \bold z)] - \mathrm{D_{KL}}(q_\phi(\bold z|\bold x)||p(\bold z))$$

To get a loss function we can work with, we now have to solve both of these terms. Lets start with the regularization one. We defined both how $\bold q_\phi(\bold z| \bold x)$ and $p(\bold z)$ (prior) look like so we can write it as:

$$\mathrm{D_{KL}}(q_\phi(\bold z|\bold x)||p(\bold z))\\=\mathrm{D_{KL}}(\mathcal N(\bm \mu,\bm \sigma^2)||\mathcal N(0,\bold I))=-\frac{1}{2}\sum_{j=1}^J(1+\log(\sigma^2_j)-\mu_j^2-\sigma^2_j)$$

The result we get by divine intervention (I copy-pasted it). Regarding the reconstruction term, it is important that we can't take a derivative of sampling, so we use the "reparametrization trick" that will ["move" sampling outside of backward pass path](./notes_misc.md#vae-reparametrization-trick) (very cool Gemini):

$$\bold z = \bm \mu_{\phi} + (\bm\sigma_{\phi} \odot \bm \epsilon)$$

Then one can rewrite the reconstruction term as (why):

$$\mathbb{E}_{\bold z \sim q_{\phi}(\bold z|\bold x)}[\log p_\theta(\bold x| \bold z)]=\mathbb{E}_{\bold \epsilon \sim \mathcal N(0,1)}[\log p_\theta(\bold x| \bm \mu_{\phi} + (\bm\sigma_{\phi} \odot \bm \epsilon))] = \int \mathcal N(\epsilon; 0,1)\log p(x| \bm\mu+\bm\sigma\odot\bm\epsilon)d\epsilon$$

While we're at it, [why is VAE using Gaussians?](./notes_misc.md#why-is-vae-using-gaussians). So the gradient of just the reconstruction term we can write as:

$$\nabla_\phi \mathbb{E}_{\bold \epsilon \sim \mathcal N(0,1)}[\log p_\theta(\bold x| \bm \mu_{\phi} + (\bm\sigma_{\phi} \odot \bm \epsilon))] = \mathbb{E}_{\bold \epsilon \sim \mathcal N(0,1)}[\nabla_{\phi}\log p_\theta(\bold x| \bm \mu_{\phi} + (\bm\sigma_{\phi} \odot \bm \epsilon))]\\\approx\frac 1 N \sum ^N _{i=1} \nabla_\phi \big(-\frac 1 2 ||x-\mathrm{Dec}(\mathrm{Enc}(x)_\mu + (\mathrm{Enc}(x)_\sigma\odot\epsilon))||^2\big) $$

<!-- $$\mathcal L_\mathrm{rec} = \frac 1 {2N} \sum ||x-\hat x||^2$$ wrong -->

Since we optimize by **Minimizing Loss** (which is equivalent to Maximizing ELBO), we flip the signs of the ELBO equation. This way both terms are now positive and we can properly minimize.

$$ \text{Loss} = -\text{ELBO} = -\text{Reconstruction} + \text{KL Divergence} $$

Given a batch of $N$ images $\{\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(N)}\}$, and latent dimension size $J$:

$$
\mathcal{L}(\theta, \phi) \approx \frac{1}{N} \sum_{i=1}^N \left( 
\underbrace{\frac{1}{2} ||\mathbf{x}^{(i)} - \mathrm{Dec}_\theta(\mathbf{z}^{(i)})||^2}_{\text{Reconstruction Loss (MSE)}} 
\;+\; 
\underbrace{\frac{1}{2} \sum_{j=1}^J \left( \mu_{i,j}^2 + \sigma_{i,j}^2 - \log(\sigma_{i,j}^2) - 1 \right)}_{\text{Regularization Loss (Analytic KL)}} 
\right)
$$

**Where:**
1.  **The Encoder Outputs:** $\boldsymbol{\mu}^{(i)}, \boldsymbol{\sigma}^{(i)} = \mathrm{Enc}_\phi(\mathbf{x}^{(i)})$
2.  **The Noise:** $\boldsymbol{\epsilon}^{(i)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
3.  **The Reparameterization:** $\mathbf{z}^{(i)} = \boldsymbol{\mu}^{(i)} + \boldsymbol{\sigma}^{(i)} \odot \boldsymbol{\epsilon}^{(i)}$

## VAE to Diffusion bridge (DDPM)

First intuition is what the forward process does (the forward process is our "encoder"/surrogate):

$$x_0 \rightarrow q(x_1|x_0) \rightarrow x_1 \rightarrow ... \rightarrow x_T$$

But the encoder is just some usually constant kernel that (usually) just adds noise in a fixed way. So now instead of creating $\bm z$ in a single step, we create a chain $\bm z_1, ... \bm z_T$, but use $\bm x$ for notation. So encoder looks something like this:

$$q(x_{t+1}| x_t)=\mathcal N(x_{t+1}; \sqrt{1-\beta}x_t, \beta\bm I) $$

And we can again use the same reparametrization trick that we came up with for VAEs:

$$\bm x_{t+1}=\sqrt{1-\beta}\bm x_t + \sqrt \beta\bm\epsilon; \bm\epsilon\sim\mathcal N(0, \bm I)$$

However papers usually use the $x_{t-1}$ notation and $\beta$ is time-dependent:

$$ q(x_t| x_{t-1})=\mathcal N(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\bm I) $$

And because a sum of Gaussians is a Gaussian, we don't have to iteratively add noise, we can instead specify how many steps we want to do at once:

$$ q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)\mathbf{I}) $$

*   $\alpha_t = 1 - \beta_t$ (How much signal we keep)
*   $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ (The cumulative signal kept from step 0 to $t$)

Now the decoder part or reverse process is:

$$x_T \rightarrow p_\theta(x_{T-1}|x_{T}) \rightarrow x_{T-1} \rightarrow ... \rightarrow x_0$$

And in this diffusion model setting we only want to learn the decoder $p_\theta(x_{t-1}|x_{t})$ part of VAE setup, since the forward process is purely algebraic. A single step in this reverse process is:

<!-- $$p_\theta(x_{t-1}|x_t)=$$ -->





### Important:
https://mbernste.github.io/posts/vae/ \
https://yunfanj.com/blog/2021/01/11/ELBO.html \
https://www.wpeebles.com/DiT