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

- We have a trained diffusion model $p_{\theta}(x_t | x_{t+1})$.
-  Question: how do we condition this model to generate a sample of a given class $y$?
  -  Example: consider that the authors of DDPM trained on ImageNet, which has 1000 classes. When we ask the model for a sample, currently we do not have a way to control the class it will generate a sample from.

- Solution: train a classifier $p_{\phi}(y | x_t) $ and use the gradient to "push" the diffusion model in the correct direction during sampling.

## Math Behind Classifier Guidance

OK, why does this work? It turns out that if we want a conditional sampling process, we can sample from the following distribution

$$
\underbrace{p_{\theta, \phi}(x_t | x_{t+1}, y)}_{\text{what we want...}} = \underbrace{Z p_{\theta}(x_t | x_{t+1})p_{\phi}(y | x_t)}_{\text{...can be factored as a product} }
$$

where $Z$ is a normalizing constant. What we have shown is that the distribution that we want to sample from, $p_{\theta, \phi}(x_t | x_{t+1}, y)$, can be factored into a product between the unconditional diffusion model we already have and a classifier trained on noisy images $x_t$.

<!-- <details>
  <summary>Derivation of this factorization</summary>

  This is the hidden text that appears when you click the summary line.  
  You can add **Markdown formatting** here too, like _italics_, **bold**, or even:

  - Bullet points
  - Code blocks
  - Images
</details> -->


