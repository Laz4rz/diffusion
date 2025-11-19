# Global surrogate in Variational Inference

*Or why it's bad to have it and we in theory need unique surrogate per data point.*

Here is the expansion of that summary. We will break it down into the **Reconstruction** failure and the **Regularization** collapse.

Let’s run a thought experiment with a very simple dataset.

### The Setup
*   **Dataset ($D$):** 5,000 images of **Black Squares** (pixel value 0) and 5,000 images of **White Squares** (pixel value 1).
*   **Your Proposal:** Learn **one global** $q(z)$ (a single Gaussian defined by $\mu, \sigma$) to represent the whole dataset.
*   **The Prior ($p(z)$):** Standard Normal $\mathcal{N}(0,1)$.

---

### 1. The Reconstruction Failure (The "Average Image")

We start training. We pick a batch of data containing both Black and White squares.

1.  **Sampling:** We sample a $z$ from your global $q(z)$. Let's say we get $z = 0.5$.
2.  **The Conflict:** Since $q(z)$ is global (it doesn't change based on the input), this value $z=0.5$ is currently being used to represent **both** a Black square AND a White square in your batch.
3.  **The Decoder's Job:** The decoder receives $z=0.5$. It has to predict a pixel value $\hat{x}$.
    *   The loss function says: "Minimize the distance to the Black Square (0)."
    *   The loss function *also* says: "Minimize the distance to the White Square (1)."
4.  **The Math:** What number minimizes the error between itself and 0, AND itself and 1?
    *   The average: **0.5 (Grey)**.

**Conclusion:** Since the latent code $z$ contains no information about *which* image you are currently trying to reconstruct (because it's global), the Decoder's best mathematical bet is to simply output the **average of the entire dataset**.

If you did this on MNIST (digits), your output would be a blurry grey blob that looks like every number superimposed on top of each other.

---

### 2. The Regularization Collapse (The "Prior")

Now look at the second term of the ELBO:
$$ - D_{KL}(q(z) || p(z)) $$

This term punishes the model if $q(z)$ is different from the standard prior (Normal distribution centered at 0).

1.  **The Trade-off:** Usually, in a VAE, we tolerate some KL penalty because $q(z|x)$ gives us useful information to help the reconstruction. We pay a "tax" (KL) to buy "information" (Reconstruction).
2.  **The Reality of Global $q$:** As we saw above, a global $q(z)$ provides **zero information** to the decoder. $z$ is just random noise uncorrelated with the specific image $x$.
3.  **The Gradient Descent:** The optimizer looks at the loss:
    *   *Reconstruction Term:* "Changing $q(z)$ doesn't help me distinguish Black from White, so I don't care what $q(z)$ is."
    *   *KL Term:* "Hey! If you make $q(z)$ exactly equal to $\mathcal{N}(0,1)$, this penalty becomes Zero!"
4.  **The Result:** The optimizer realizes the latent code is useless for reconstruction, so it simply gives up on encoding and sets $q(z) = p(z)$ to minimize the KL term perfectly.

---

### Intuitive Summary
If $q(z)$ is global, you are effectively cutting the wire between the Input Image and the Latent Code.

*   **Input:** Image of a Dog.
*   **Global $q(z)$:** "I don't care that it's a dog. Here is a random number: 0.42."
*   **Decoder:** "What image corresponds to 0.42? Well, during training, 0.42 was paired with dogs, cats, cars, and planes. I will output a blurry grey mess."

**To fix this:**
You need the wire connected. You need $q$ to change based on $x$.
*   **Input:** Image of a Dog.
*   **Local $q(z|x)$:** "Oh, a dog! I will give you a number from the 'Dog Cluster', let's say 5.2."
*   **Decoder:** "Ah, 5.2. I know that region is usually dogs. Here is a picture of a dog."
