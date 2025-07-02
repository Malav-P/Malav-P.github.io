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

This is a tractable objective because we can estimate it by sampling from $q_{\phi}(z|x)$. Note that maximizing the ELBO does two things:

1. Increases the log probability of observed samples (i.e. we are learning a good latent variable model that represents the data well)
2. Reduces the "distance" between the variational distribution and the true posterior (i.e. we are learning a good approximate posterior distribution)
