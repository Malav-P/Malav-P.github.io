---
layout: post
title: "Classifier Guidance"
katex: True
blurb: ""
img: ""
author: "Malav Patel"
categories: journal
tags: []
<!-- image: -->
---


## Classifier Guidance
- We have a trained diffusion model $p_{\theta}(x_t \mid x_{t+1})$.
-  Question: how do we condition this model to generate a sample of a given class $y$?
  -  Example: consider that the authors of DDPM trained on ImageNet, which has 1000 classes. When we ask the model for a sample, currently we do not have a way to control the class it will generate a sample from.

- Solution: train a classifier $p_{\phi}(y \mid x_t) $ and use the gradient to "push" the diffusion model in the correct direction during sampling.

## Math Behind Classifier Guidance

OK, why does this work? It turns out that if we want a conditional sampling process, we can sample from the following distribution

$$
\underbrace{p_{\theta, \phi}(x_t \mid x_{t+1}, y)}_{\text{what we want...}} = \underbrace{Z p_{\theta}(x_t \mid x_{t+1})p_{\phi}(y \mid x_t)}_{\text{...can be factored as a product} }
$$

where $Z$ is a normalizing constant. What we have shown is that the distribution that we want to sample from, $p_{\theta, \phi}(x_t \mid x_{t+1}, y)$, can be factored into a product between the unconditional diffusion model we already have and a classifier trained on noisy images $x_t$.

<details>
  <summary>Derivation of this factorization</summary>

 **Goal**: Show that $p_{\theta, \phi}(x_t \mid x_{t+1}, y) = Z p_{\theta}(x_t \mid x_{t+1})p_{\phi}(y \mid x_t)$.

 Begin by defining the conditional joint distribution $\hat{q}$: 

$$
\begin{aligned}
\underbrace{\hat{q}(x_{t+1} \mid x_t, y)}_{\substack{\text{conditional forward} \\ \text{process is...}}}
&:=
\underbrace{q(x_{t+1} \mid x_t)}_{\substack{\text{... the same as our} \\ \text{original process}}} \\ 

\underbrace{\hat{q}(x_0)}_{\substack{\text{marginal distribution of} \\ \text{data variable is...}}} &:= \underbrace{q(x_0)}_{\substack{\text{...the same as our original} \\ \text{ marginal distribution}}} \\ 

\hat{q}(y \mid x_0) &:= \text{known} \\ 

\hat{q}(x_{1:T} \mid x_0, y) &:= \prod_{t=0}^{T-1} \hat{q}(x_{t+1} \mid x_t, y)

\end{aligned}
$$

Note that with these definitions, we have fully defined the conditional distribution of all variables:

$$
\begin{aligned}
\hat{q}(x_{0:T} \mid y) &= \hat{q}(x_{1:T} \mid x_0, y) \hat{q}(x_0 \mid y)
\end{aligned}
$$

First we show that the defined noising process $\hat{q}$ when not conditioned on $y$ is actually the same as $q$.

$$
\begin{aligned}
\hat{q}(x_{t+1}\mid x_t) &= \int_y \hat{q}(x_{t+1}, y \mid x_t) \ dy \\ 
&= \int_y \hat{q}(x_{t+1} \mid y, x_t) \hat{q}(y\mid x_t)\ dy \\ 
&= \int_y q(x_{t+1}\mid x_t)\hat{q}(y\mid x_t)\ dy \\ 
&= q(x_{t+1}\mid x_t) \int_y \hat{q}(y\mid x_t)\ dy \\ 
&= q(x_{t+1}\mid x_t) \\ 
&= \hat{q}(x_{t+1} \mid x_t, y)

\end{aligned}
$$

Next, we do something similar for the joint distribution $\hat{q}(x_{1:T} \mid x_0)$:

$$
\begin{aligned}
\hat{q}(x_{1:T}\mid x_0) &= \int_y \hat{q}(x_{1:T}, y\mid x_0) \ dy \\ 
&= \int_y  \hat{q}(x_{1:T}\mid y, x_0) \hat{q}(y\mid x_0) \ dy \\ 
&= \int_y \hat{q}(y \mid x_0) \prod_{t=0}^{T-1}q(x_{t+1}\mid x_t) \ dy \\ 
&= \prod_{t=0}^{T-1}q(x_{t+1}\mid x_t) \int_y \hat{q}(y \mid x_0) \ dy \\ 
&= q(x_{1:T} \mid x_0)
\end{aligned}
$$

Then, we can do something similar for the marginal distribution $\hat{q}(x_t)$:

$$
\begin{aligned}
\hat{q}(x_t) &= \int_{x_{0:t-1}} \hat{q}(x_0, ..., x_t ) \ dx_{0:t-1} \\
&= \int_{x_{0:t-1}} \hat{q}(x_0) \hat{q}(x_1, ..., x_t \mid x_0) \ dx_{0:t-1} \\ 
&= \int_{x_{0:t-1}} q(x_0) q(x_1, ..., x_t \mid x_0) \ dx_{0:t-1} \\
&= \int_{x_{0:t-1}} q(x_0, ..., x_t ) \ dx_{0:t-1} \\ 
&= q(x_t)
\end{aligned}
$$

We've shown so far that $\hat{q}(x_0) = q(x_0)$ and $\hat{q}(x_{t+1} \mid x_t) = q(x_{t+1}\mid x_t)$. Using Bayes rule:

$$
\begin{aligned}
\hat{q}(x_t \mid x_{t+1}) &= \frac{\hat{q}(x_{t+1} \mid x_t)\hat{q}(x_t)}{\hat{q}(x_{t+1})} \\ 
&= \frac{q(x_{t+1} \mid x_t)q(x_t)}{q(x_{t+1})} \\ 
&= q(x_t\mid x_{t+1})
\end{aligned}
$$

Now, we show that the classifier $\hat{q}(y\mid x_t, x_{t+1})$ is actually not dependent on $x_{t+1}$:

$$
\begin{aligned}
\hat{q}(y\mid x_t, x_{t+1}) &= \hat{q}(x_{t+1}\mid y, x_t)\frac{\hat{q}(y \mid x_t)}{\hat{q}(x_{t+1} \mid x_t)} \\ 
&= \hat{q}(x_{t+1} \mid x_t)\frac{\hat{q}(y \mid x_t)}{\hat{q}(x_{t+1} \mid x_t)}\\ 
&= \hat{q}(y\mid x_t)
\end{aligned}
$$

Finally, we are ready to derive the reverse conditional process:

$$
\begin{aligned}
\hat{q}(x_t \mid x_{t+1}, y) &= \frac{\hat{q}(x_t, x_{t+1}, y)}{\hat{q}(x_{t+1}, y)}\\
&= \frac{\hat{q}(y \mid x_t, x_{t+1}) \hat{q}(x_t\mid x_{t+1})\hat{q}(x_{t+1})}{\hat{q}(y\mid x_{t+1})\hat{q}(x_{t+1})} \\ 
&= \frac{\hat{q}(x_t\mid x_{t+1})\hat{q}(y\mid x_t)}{\hat{q}(y\mid x_{t+1})} \\ 
&= \frac{q(x_t\mid x_{t+1})\hat{q}(y\mid x_t)}{\hat{q}(y\mid x_{t+1})}

\end{aligned}
$$

The denominator is constant since it does not depend on $x_t$. So, we want to sample from the distribution $Zq(x_t\mid x_{t+1})\hat{q}(y\mid x_t)$, where $Z$ is a normalization constant. A model already exists for $q(x_t\mid x_{t+1})$, called $p_{\theta}(x_t \mid x_{t+1})$. What remains is to approximate $\hat{q}(y\mid x_t)$ with a model $p_{\phi}(y \mid x_t)$. This results in a model for the conditional reverse process:

$$
p_{\theta, \phi}(x_t \mid x_{t+1}, y) = Zp_{\theta}(x_t \mid x_{t+1})p_{\phi}(y \mid x_t)
$$
</details>

Typically, is is intractable to sample from $p_{\theta, \phi}(x_t \mid x_{t+1}, y)$. To get around this, let us recall our model:

$$
\log p_{\theta}(x_t \mid x_{t+1}) = -\frac{1}{2}(x_t - \mu)^\top \Sigma^{-1}(x_t-\mu) + C
$$

Note that in the limit of infinite diffusion steps, $\|\Sigma\| \to 0$. So, we can reasonably assume that $\log p_{\phi}(y \mid x_t)$ has low curvature (2nd order characteristics) compared to $\log p_{\theta}(x_t \mid x_{t+1})$. So we consider the  taylor expansion of $\log p_{\phi}(y \mid x_t)$ to first order around $x_t = \mu$:

$$
\begin{aligned}
\log p_{\phi}(y \mid x_t) &\approx \log p_{\phi}(y \mid x_t)\mid_{x_t=\mu} + (x_t-\mu)^\top \nabla_{x_t} \log p_{\phi}(y\mid x_t) \mid_{x_t = \mu} \\ 
&= \log p_{\phi}(y \mid x_t)\mid_{x_t=\mu} + (x_t-\mu)^\top g
\end{aligned}
$$

Where $g = \nabla_{x_t} \log p_{\phi}(y\mid x_t) \mid_{x_t = \mu}$ is the score function of the classifier evaluated at $\mu$.

Now let's consider our model for the reverse conditional process and approximate it using only the first order expansion of $\log p_{\phi}$. Remember that in comparison, $\log p_{\theta}$ has much higher curvature, so we are not losing much information by truncating to first order. The reverse conditional process can be approximated as follows:

$$
\begin{aligned}
\log p_{\theta, \phi}(x_t \mid x_{t+1}, y) &= \log p_{\theta}(x_t\mid x_{t+1}) + \log p_{\phi}(y \mid x_t) + C_1 \\ 
&= -\frac{1}{2}(x_t - \mu)^\top \Sigma^{-1}(x_t-\mu) + \log p_{\phi}(y \mid x_t) + C_1 \\ 
&\approx -\frac{1}{2}(x_t - \mu)^\top \Sigma^{-1}(x_t-\mu) + \underbrace{\log p_{\phi}(y \mid x_t)\mid_{x_t=\mu}}_{\text{constant w.r.t }x_t} + (x_t-\mu)^\top g + C_1 \\ 
&= -\frac{1}{2}(x_t - \mu - \Sigma g)^\top \Sigma^{-1}(x_t-\mu - \Sigma g) + \underbrace{\frac{1}{2}g^\top \Sigma g}_{\text{constant w.r.t }x_t} + C_2 \\
&= -\frac{1}{2}(x_t - \mu - \Sigma g)^\top \Sigma^{-1}(x_t-\mu - \Sigma g) + C_3 \\ 
&= \log p(z) + C_4, \  z \sim \mathcal{N}(\mu + \Sigma g, \Sigma)

\end{aligned}
$$

Recall that our unconditional reverse process was $p_{\theta}(x_t \mid x_{t+1}) = \mathcal{N}(\mu, \Sigma)$.
What we have just shown is that we can sample from $p_{\theta, \phi}(x_t \mid x_{t+1}, y)$ approximately by sampling from the unconditional reverse process but with a shifted mean $\mu + \Sigma g$ and variance $\Sigma$. 

In this sense: by **"classifier guidance"** we mean that the classifier "guides" the original unconditional reverse process by shifting its mean by an amount proportional to the gradient of the classifier.

## References

[1] Dhariwal, Prafulla, and Alexander Nichol. "Diffusion models beat gans on image synthesis." Advances in neural information processing systems 34 (2021): 8780-8794.
