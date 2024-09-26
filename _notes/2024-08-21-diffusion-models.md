---
layout: post
title: "Diffusion Models"
katex: True
blurb: ""
img: ""
author: "Malav Patel"
categories: journal
tags: []
<!-- image: -->
---

In this note, we describe diffusion models.

Diffusion models are latent variable models of the form $p_{\theta}(\mathbf{x}_0) = \int p_{\theta}(\mathbf{x}_{0:T})d\mathbf{x}_{1:T}$. Note that this expression is simply the marginal distribution of $\mathbf{x}_0$ assuming a joint distribution $p_{\theta}(\mathbf{x}_{0:T})$ exists. The marginal distribution can be found by integrating the remaining variables $\mathbf{x}_1, \mathbf{x}_2\ldots \mathbf{x}_T$ out of the joint distribution.

For diffusion models, the joint distribution takes the following form,
$$
\begin{equation}
p_{\theta}(\mathbf{x}_{0:T}) = p(\mathbf{x}_T)\prod\limits_{t=1}^{T} p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_t)
\end{equation}
$$

## Reverse Process

Note that the joint disribution is constructed by an iterated product of conditional distributions, starting with $p(\mathbf{x}_T)$. Usually, diffusion models assume the following forms for the terms in Eq $(1)$:
$$
\begin{align*}
p(\mathbf{x}_T) &= \mathcal{N}(\mathbf{0}, \mathbf{I})\\ 
p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_t) &= \mathcal{N}\big(\mathbf{x}_{t-1}; \boldsymbol{\mu}_{\theta}(\mathbf{x}_t, t), \boldsymbol{\Sigma}_{\theta}(\mathbf{x}_t, t)\big)
\end{align*}
$$ 

To get a sample $x \sim p_{\theta}(\mathbf{x}_0)$, we start by sampling from $p(\mathbf{x}_T)$ and then sample each of the conditional distributions in a chain. This is referred to as the reverse process. A diagram is shown below.

<center>
<figure>
  <img src="../assets/img/diffusion_0.png" width="95%">
  <figcaption><i>The reverse process in diffusion</i></figcaption>
</figure>
</center>

## Forward Process
In diffusion models, the forward process is fixed to a markov chain. The transition between two successive states is defined by a gaussian distribution.

$$
\begin{aligned}
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\mathbf{x}_{t-1}; \beta_t\mathbf{I})
\end{aligned}
$$

The collection $\{\beta_t\}_{t=1}^{T}$ defines the $\it{\text{variance schedule}}$ of the diffusion process and is often fixed (not learned via training).



# Diffusion from A Gradient Matching Perspective

Consider the data distribution $p(\mathbf{x})$. Let us introduce a surrogate model of the form $p(\mathbf{x}' | \mathbf{x}) := \mathcal{N}(\sqrt{1-\beta}\mathbf{x}, \beta \mathbf{I})$. This is equivalent to saying 
$$
\mathbf{x}' = \sqrt{1-\beta}\mathbf{x} + \sqrt{\beta}\mathbf{\epsilon} \quad \mathbf{\epsilon} \sim \mathcal{N(0, \mathbf{I})}
$$

The distribution over $\mathbf{x}'$ can be found with the chain rule of probability,
$$
p(\mathbf{x}') = \int p(\mathbf{x}) p(\mathbf{x}' | \mathbf{x}) \ d\mathbf{x}
$$

Now consider the gradient of the score function for this distribution,
$$
\begin{aligned}
\nabla_{\mathbf{x}'}\log{p(\mathbf{x}')} &= \frac{\nabla_{\mathbf{x}'}\int p(\mathbf{x})p(\mathbf{x}' | \mathbf{x}) \ d\mathbf{x}}{p(\mathbf{x}')} \\ 
&= \frac{\int p(\mathbf{x}) \nabla_{\mathbf{x}'}p(\mathbf{x}' | \mathbf{x}) \ d\mathbf{x}}{p(\mathbf{x}')} \\ 
&= \int \frac{p(\mathbf{x})p(\mathbf{x}'|\mathbf{x})}{p(\mathbf{x}')}\nabla_{\mathbf{x}'}\log{p(\mathbf{x}'|\mathbf{x})} \ d\mathbf{x} \\ 
&= \int p(\mathbf{x} | \mathbf{x}') \nabla_{\mathbf{x}'}\log{p(\mathbf{x}'|\mathbf{x})} \ d\mathbf{x} \\ 
&= \mathbb{E}_{\mathbf{x}|\mathbf{x}'}[\nabla_{\mathbf{x}'}\log{p(\mathbf{x}'|\mathbf{x})}]
\end{aligned}
$$

Let us break down the term inside the expectation.
$$
\begin{aligned}
\nabla_{\mathbf{x}'}\log{p(\mathbf{x}'|\mathbf{x})} &= - \nabla_{\mathbf{x}'}\frac{\|\mathbf{x}' - \sqrt{1-\beta}\mathbf{x} \|^2}{2\beta} \\ 
&= \frac{\sqrt{1-\beta}\mathbf{x}-\mathbf{x}'}{\beta}
\end{aligned}
$$

Returning to the beginning we have,

$$
\begin{aligned}
\nabla_{\mathbf{x}'}\log{p(\mathbf{x}')} &= \mathbb{E}_{\mathbf{x}'|\mathbf{x}}[\nabla_{\mathbf{x}'}\log{p(\mathbf{x}'|\mathbf{x})}] \\ 
&= \mathbb{E}_{\mathbf{x}|\mathbf{x}'}\bigg[\frac{\sqrt{1-\beta}\mathbf{x} - \mathbf{x}'}{\beta}\bigg]
\end{aligned}
$$

Rearranging terms outside the expectation,

$$
\begin{equation}
\mathbf{x}' + \beta\nabla_{\mathbf{x}'}\log{p(\mathbf{x}')} = \sqrt{1-\beta}\mathbb{E}_{\mathbf{x}|\mathbf{x}'}[\mathbf{x}]
\end{equation}
$$

Equation (2) tell us we can find the expected value of $\mathbf{x}$ given $\mathbf{x}'$. However, we cannot calculate the score function for $p(\mathbf{x}')$. Instead, let us use a function approximator to approximate the score function,

$$
s_{\theta}(\mathbf{x}', \beta) \approx \nabla_{\mathbf{x}'}\log{p(\mathbf{x}')}

$$

Then we can use (2) to define a minimization objective:

$$
\begin{equation}
\min_\theta \quad \mathbb{E}_{\beta}\mathbb{E}_{\mathbf{x}, \mathbf{x}'}[\|\mathbf{x}' + \beta s_{\theta}(\mathbf{x}', \beta) - \sqrt{1-\beta}\mathbb{E}_{\mathbf{x}'|\mathbf{x}}[\mathbf{x}]\|^2]
\end{equation}
$$

The expectation inside of the loss function makes optimization difficult. Let us derive an equivalent objective by considering the expectation over $\mathbf{x}'$ and $\mathbf{x}$:

$$
\begin{aligned}
\mathbb{E}_{\mathbf{x}, \mathbf{x}'}[\|\mathbf{x}' + \beta s_{\theta}(\mathbf{x}', \beta) - \sqrt{1-\beta}\mathbb{E}_{\mathbf{x}|\mathbf{x}'}[\mathbf{x}]\|^2] &= \mathbb{E}_{\mathbf{x},\mathbf{x}'}[\|\mathbf{x}' + \beta s_{\theta}(\mathbf{x}', \beta) - \sqrt{1-\beta}\mathbf{x} + \sqrt{1-\beta}\mathbf{x} - \sqrt{1-\beta}\mathbb{E}_{\mathbf{x}|\mathbf{x}'}[\mathbf{x}]\|^2] \\ 
&= \mathbb{E}_{\mathbf{x},\mathbf{x}'}[\|\mathbf{x}' + \beta s_{\theta}(\mathbf{x}', \beta) - \sqrt{1-\beta}\mathbf{x}\|^2] + \underbrace{(1-\beta)\mathbb{E}_{\mathbf{x},\mathbf{x}'}[\|\mathbf{x} - \mathbb{E}_{\mathbf{x}|\mathbf{x}'}[\mathbf{x}]\|^2]}_{\text{independent of}\ \  \theta} + 2\mathbb{E}_{\mathbf{x},\mathbf{x}'}[\langle \mathbf{x}' + \beta s_{\theta}(\mathbf{x}', \beta) - \sqrt{1-\beta}\mathbf{x}, \sqrt{1-\beta}\mathbf{x} - \sqrt{1-\beta}\mathbb{E}_{\mathbf{x}|\mathbf{x}'}[\mathbf{x}]\rangle]
\end{aligned}
$$

Note that the second term is independent of $\theta$ so it can be dropped from the loss function as it is a constant. Consider the inner product from the third term.

$$
\begin{aligned}
\mathbb{E}_{\mathbf{x},\mathbf{x}'}[\langle \mathbf{x}' + \beta s_{\theta}(\mathbf{x}', \beta) - \sqrt{1-\beta}\mathbf{x}, \sqrt{1-\beta}\mathbf{x} - \sqrt{1-\beta}\mathbb{E}_{\mathbf{x}|\mathbf{x}'}[\mathbf{x}]\rangle] &= \mathbb{E}_{\mathbf{x}'}\mathbb{E}_{\mathbf{x}|\mathbf{x}'}[\langle \mathbf{x}' + \beta s_{\theta}(\mathbf{x}', \beta) - \sqrt{1-\beta}\mathbf{x}, \sqrt{1-\beta}\mathbf{x} - \sqrt{1-\beta}\mathbb{E}_{\mathbf{x}|\mathbf{x}'}[\mathbf{x}]\rangle]\\ 
&= \mathbb{E}_{\mathbf{x}'}\bigg[\bigg\langle \mathbf{x}' + \beta s_{\theta}(\mathbf{x}', \beta) - \sqrt{1-\beta}\mathbb{E}_{\mathbf{x}|\mathbf{x}'}[\mathbf{x}], \underbrace{\sqrt{1-\beta}\mathbb{E}_{\mathbf{x}|\mathbf{x}'}[\mathbf{x}] - \sqrt{1-\beta}\mathbb{E}_{\mathbf{x}|\mathbf{x}'}[\mathbf{x}]}_{=\ 0}\bigg\rangle\bigg] \\ 
&= 0
\end{aligned}
$$

Now we can rewrite our objective as 
$$
\min_{\theta} \quad \mathbb{E}_{\beta}\mathbb{E}_{\mathbf{x},\mathbf{x}'}[\|\mathbf{x}' + \beta s_{\theta}(\mathbf{x}', \beta) - \sqrt{1-\beta}\mathbf{x}\|^2]
$$

Recall that $\mathbf{x}' - \sqrt{1-\beta}\mathbf{x} = \sqrt{\beta}\mathbf{\epsilon}$ and let us modify the approximator slightly like so: $s_{\theta}(\mathbf{x}', \beta) \approx -\frac{1}{\beta}\nabla_{\mathbf{x}'}\log{p(\mathbf{x}')}$. Then our objective can be written as

$$
\begin{equation}
\min_{\theta} \quad \mathbb{E}_{\beta}\mathbb{E}_{\mathbf{x},\mathbf{x}'}[\beta\|s_{\theta}(\mathbf{x}', \beta) - \mathbf{\epsilon}\|^2]
\end{equation}
$$

