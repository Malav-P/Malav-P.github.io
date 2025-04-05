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
</br>

 Goal: Show that $p_{\theta, \phi}(x_t \mid x_{t+1}, y) = Z p_{\theta}(x_t \mid x_{t+1})p_{\phi}(y \mid x_t)$.

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




</details>


