---
layout: post
title: "Variational Inference"
katex: True
blurb: ""
img: ""
author: "Malav Patel"
categories: journal
tags: []
<!-- image: -->
---

What is it? A method for approximating the posterior distribution in latent variable models.

What do we need it for? If we want to learn the latent variable model given a dataset of observations, or if we want to find the posterior distribution.


### Problem setup

Want to optimize

$$
\log p_{\theta}(x) = \log \int p_{\theta}(x, z) dz
$$

But the integral is usually intractable.

### Solution

Use the ELBO

$$
\log p_{\theta}(x) - \text{KL}[q_{\phi}(z|x)\|p_{\theta}(z|x)] = \underbrace{ \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)] - \text{KL}[q_{\phi}(z|x) \| p(z)]}_{\text{ELBO}}
$$

This is a tractable objective because we can estimate it by sampling from $q_{\phi}(z\|x)$. Note that maximizing the ELBO does two things:

1. Increases the log probability of observed samples (i.e. we are learning a good latent variable model that represents the data well)
2. Reduces the "distance" between the variational distribution and the true posterior (i.e. we are learning a good approximate posterior distribution)


### Learning the Model
A simple algorithm would be


- x $\gets$ get datapoint from dataset
- sample $z \sim q_{\phi}(\cdot \|x)$ several times
- use set of sampled $z$ to get Monte Carlo Estimate of ELBO, call it $\mathcal{L}$.
- Take gradient of ELBO, $\nabla_{\theta}\mathcal{L}$ and $ \nabla_{\phi}\mathcal{L}$
- Take a gradient step
   - $\theta \gets \theta + \nabla_{\theta}\mathcal{L}$
   - $\phi \gets \phi + \nabla_{\phi}\mathcal{L}$
- Repeat until converged


To get less noisy estimates of the gradient, we would normally do minibatches of $x$.

### After We've Learned
After we have learned [1], we can generate samples by doing the following:

- sample from our prior over the latent variable $z \sim p(z)$.
- sample from our likelihood model $x \sim p_{\theta}(x\|z)$.


[1]: Note that most latent variable models are parameterized as 
$$
p_{\theta}(x, z) = p_{\theta}(x | z)p(z)
$$
That is, we have a known prior $p(z)$ over the latent variable that is not learned. Typically it is a tractable distribution like a Gaussian.




