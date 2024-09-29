---
layout: post
title: "Score Matching"
katex: True
blurb: ""
img: ""
author: "Malav Patel"
categories: journal
tags: []
<!-- image: -->
---

<!-- In this note, we describe score matching.

Consider a dataset $\mathcal{D} = \{\mathbf{x}\_i\}\\_{i=1}^{n}$ where $\mathbf{x} \in \mathbb{R}^d$. Consider an energy based model

$$
p(\mathbf{x}) = \frac{e^{-f_{\theta}(x)}}{Z(\theta)}
$$

Suppose we optimize parameters $\theta$ using maximum likelihood.

$$
\begin{aligned}
l(\theta) &= \frac{1}{n}\sum_{i=1}^{n}\log p(\mathbf{x}_i) \\ 
&=  - \log Z(\theta) - \frac{1}{n}\sum_{i=1}^{n} f_{\theta}(\mathbf{x}_i)
\end{aligned}
$$

Taking the gradient,

$$
\begin{aligned}
\nabla_{\theta}\ l(\theta) &= - \nabla_{\theta}\log Z(\theta) -\frac{1}{n}\sum_{i=1}^{n} \nabla_{\theta}f_{\theta}(\mathbf{x}_i)  \\ 
&=  \mathbb{E}_{p(\mathbf{x})}[\nabla_{\theta}f_{\theta}(\mathbf{x})]  -\frac{1}{n}\sum_{i=1}^{n} \nabla_{\theta}f_{\theta}(\mathbf{x}_i) 
\end{aligned}
$$ -->

## Score Matching
Suppose we do not wish to work with the intractable $Z(\theta)$ that arises when trying to optimize energy based models. Consider then the problem of finding a distribution $p$  that minimizes the Fisher divergence between our data distribution $\hat{p}$ and the model distribution $p$:

$$
\begin{aligned}
\min_{p} \quad D_F(p \ ||\  \hat{p}) :&= \mathbb{E}_{\hat{p}(\mathbf{x})}[ \| \nabla_{\mathbf{x}}\log \hat{p}(\mathbf{x}) - \nabla_{\mathbf{x}
}\log p(\mathbf{x})  \|^2] \\ 
&= \mathbb{E}_{\hat{p}(\mathbf{x})}[\|\nabla_{\mathbf{x}}\log \hat{p}(\mathbf{x})\|^2] + \mathbb{E}_{\hat{p}(\mathbf{x})}[\|\nabla_{\mathbf{x}}\log p(\mathbf{x})\|^2 ] - 2 \mathbb{E}_{\hat{p}(\mathbf{x})}[\langle \nabla_{\mathbf{x}}\log \hat{p}(\mathbf{x}), \nabla_{\mathbf{x}
}\log p(\mathbf{x}) \rangle]
\end{aligned}
$$

The first term is constant with respect to $p$ and can be dropped during optimization. The second term is dependent on $p$ and can be sampled. Looking at the last term:

$$
\begin{aligned}
\mathbb{E}_{\hat{p}(\mathbf{x})}[\langle \nabla_{\mathbf{x}}\log \hat{p}(\mathbf{x}), \nabla_{\mathbf{x}
}\log p(\mathbf{x}) \rangle] &= \int\hat{p}(\mathbf{x})\langle \nabla_{\mathbf{x}}\log \hat{p}(\mathbf{x}), \nabla_{\mathbf{x}
}\log p(\mathbf{x}) \rangle\  d\mathbf{x} \\ 
&= \int\hat{p}(\mathbf{x})\bigg\langle \frac{\nabla_{\mathbf{x}}\  \hat{p}(\mathbf{x})}{\hat{p}(\mathbf{x})}, \nabla_{\mathbf{x}
}\log p(\mathbf{x}) \bigg\rangle\  d\mathbf{x}\\ 
&= \int \langle \nabla_{\mathbf{x}}\ \hat{p}(\mathbf{x}), \nabla_{\mathbf{x}}\log p(\mathbf{x})\rangle \ d\mathbf{x} \\ 
&= \int_{\Gamma}\hat{p}(\mathbf{x})\langle \nabla_{\mathbf{x}}\log p(\mathbf{x}), \hat{\mathbf{n}}  \rangle\ d\Gamma - \int \hat{p}(\mathbf{x})\nabla_{\mathbf{x}}^2 \log{p(\mathbf{x})}\ d\mathbf{x} \\ 
&= - \int \hat{p}(\mathbf{x})\nabla_{\mathbf{x}}^2 \log{p(\mathbf{x})}\ d\mathbf{x} \\ 
&= -\mathbb{E}_{\hat{p}(\mathbf{x})}[\nabla_{\mathbf{x}}^2 \log{p(\mathbf{x})}]
\end{aligned}
$$

Where in line (3) to line (4) we use Green's identity with :

- $\Gamma$ understood to be the boundary of the ball of radius $r$ in $\mathbb{R}^d$ and we are considering the limit as $r \to \infty$
- $\hat{\mathbf{n}}$ is the outward unit normal of the ball.
- $\nabla_{\mathbf{x}}^2$ is the Laplace operator with respect to variable $\mathbf{x}$.

And in lines (4) to (5) we assume that one or both of the following hold:

- $\hat{p}(\mathbf{x})$ vanishes at $\infty$
- $\nabla_{\mathbf{x}}\log{p(\mathbf{x})}$ vanishes at $\infty$ 

Thus our score matching objective is

$$
\begin{aligned}
\min_{p}\quad\mathbb{E}_{\hat{p}(\mathbf{x})}[\|\nabla_{\mathbf{x}}\log p(\mathbf{x})\|^2  + 2\ \nabla_{\mathbf{x}}^2 \log{p(\mathbf{x})}]
\end{aligned}
$$

Instead of searching for the distribution $p$, we can instead search for the score function directly. More formally, we let $\mathbf{s}\_{\theta}(\mathbf{x}) = \nabla\_{\mathbf{x}}\log{p(\mathbf{x})}$ and optimize over $\theta$ instead:

$$
\min_{\theta}\quad  \mathbb{E}_{\hat{p}(\mathbf{x})}[\|\mathbf{s}_{\theta}(\mathbf{x})\|^2 + 2\ \text{tr}\big(\nabla_{\mathbf{x}}\mathbf{s}_{\theta}(\mathbf{x}\big))]
$$

Now with a model of the score function, we can generate samples from the data distribution using Langevin dynamics:

1. Sample $\mathbf{x}_0 \sim \pi(\mathbf{x})$ from some prior distribution $\pi$.
2. Recursively compute $\mathbf{x}\_t = \mathbf{x}\_{t-1} + \frac{\epsilon}{2}\mathbf{s}\_{\theta}(\mathbf{x}\_{t-1}) + \sqrt{\epsilon}\ \mathbf{z}\_t$, where $\mathbf{z}\_t \sim \mathcal{N}(0, I)$ and some small $\epsilon > 0$.

For large number of recursive iterations, the samples $\mathbf{x}_t$ will start to converge to samples from the true distribution (under some regularity conditions, which for practical purposes are often ignored). 

# Denoising Score Matching
Suppose that materializing the jacobian $\nabla\_{\mathbf{x}}\mathbf{s}\_{\theta}(\mathbf{x})$ becomes difficult. Let us introduce another model $q(\mathbf{x}'|\mathbf{x})$ that perturbs the data with noise. Then $q(\mathbf{x}') = \int p(\mathbf{x})q(\mathbf{x}'|\mathbf{x})\ d\mathbf{x} $. The intuition is that if the noise is small enough, then the score function of the perturbed distribution will be approximately equal to that of the true distribution. Then we can run Langevin dynamics on this perturbed distribution. Our objective is

$$
\begin{aligned}
J(\theta) &= \mathbb{E}_{\mathbf{x}'}[ \| \nabla_{\mathbf{x}'}\log q(\mathbf{x}') - \mathbf{s}_{\theta}(\mathbf{x}') \|^2] \\ 
&= \mathbb{E}_{\mathbf{x}'}[\|\mathbf{s}_{\theta}(\mathbf{x}') \|^2] - 2\mathbb{E}_{\mathbf{x}'}[\langle \nabla_{\mathbf{x}'}\log q(\mathbf{x}'), \mathbf{s}_{\theta}(\mathbf{x}')  \rangle ] + \underbrace{\mathbb{E}_{\mathbf{x}'}[\|\nabla_{\mathbf{x}'}\log q(\mathbf{x}')\|^2]}_{\text{independent of \  $\theta$}}
\end{aligned}
$$

Now let us take a look at the second term in this sum,

$$
\begin{aligned}
\mathbb{E}_{\mathbf{x}'}[\langle \nabla_{\mathbf{x}'}\log q(\mathbf{x}'), \ \mathbf{s}_{\theta}(\mathbf{x}')  \rangle ] &= \int q(\mathbf{x}') \langle \nabla_{\mathbf{x}'}\log q(\mathbf{x}'),\  \mathbf{s}_{\theta}(\mathbf{x}')\rangle\  d\mathbf{x}' \\ &= \int q(\mathbf{x}')\Big\langle \frac{\nabla_{\mathbf{x}'}q(\mathbf{x}')}{q(\mathbf{x}')},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\ d\mathbf{x}' \\ 
&= \int \Big\langle \nabla_{\mathbf{x}'}q(\mathbf{x}'),\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\ d\mathbf{x}' \\ 
&= \int \Big\langle \nabla_{\mathbf{x}'}\int p(\mathbf{x})q(\mathbf{x}'|\mathbf{x})\ d\mathbf{x},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\ d\mathbf{x}' \\ 
&= \int \Big\langle \int p(\mathbf{x})\nabla_{\mathbf{x}'}q(\mathbf{x}'|\mathbf{x})\ d\mathbf{x},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\ d\mathbf{x}' \\ 
&= \int \Big\langle \int p(\mathbf{x})q(\mathbf{x}'|\mathbf{x})\nabla_{\mathbf{x}'}\log{q(\mathbf{x}'|\mathbf{x})}\ d\mathbf{x},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\ d\mathbf{x}' \\ 
&= \int \int p(\mathbf{x})q(\mathbf{x}'|\mathbf{x})\Big\langle \nabla_{\mathbf{x}'}\log{q(\mathbf{x}'|\mathbf{x})},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\ d\mathbf{x}'d\mathbf{x} \\ 
&= \mathbb{E}_{\mathbf{x},\mathbf{x}'}\Big[\Big\langle \nabla_{\mathbf{x}'}\log{q(\mathbf{x}'|\mathbf{x})},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\Big]
\end{aligned}
$$

So our score matching objective is 
$$
\begin{aligned}
J(\theta) &= \mathbb{E}_{\mathbf{x}'}[\|\mathbf{s}_{\theta}(\mathbf{x}') \|^2] - 2\mathbb{E}_{\mathbf{x}'}[\langle \nabla_{\mathbf{x}'}\log q(\mathbf{x}'), \mathbf{s}_{\theta}(\mathbf{x}')  \rangle ] + \underbrace{\mathbb{E}_{\mathbf{x}'}[\|\nabla_{\mathbf{x}'}\log q(\mathbf{x}')\|^2]}_{\text{independent of \  $\theta$}} \\ 
&= \mathbb{E}_{\mathbf{x}'}[\|\mathbf{s}_{\theta}(\mathbf{x}') \|^2] - 2\mathbb{E}_{\mathbf{x},\mathbf{x}'}\Big[\Big\langle \nabla_{\mathbf{x}'}\log{q(\mathbf{x}'|\mathbf{x})},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\Big] + \underbrace{\mathbb{E}_{\mathbf{x}'}[\|\nabla_{\mathbf{x}'}\log q(\mathbf{x}')\|^2]}_{\text{independent of \  $\theta$}}
\end{aligned}
$$

In practice, we cannot estimate the first term in this objective because the marginal distribution $q(\mathbf{x}')$ is unavailable. We show that another objective is equivalent (the argmin of both objectives is the same). Consider the new objective,

$$
\begin{aligned}
\bar{J}(\theta) :&= \mathbb{E}_{\mathbf{x},\mathbf{x}'}[ \| \nabla_{\mathbf{x}'}\log q(\mathbf{x}'|\mathbf{x}) - \mathbf{s}_{\theta}(\mathbf{x}') \|^2] \\ 
&= \mathbb{E}_{\mathbf{x},\mathbf{x}'}[\|\mathbf{s}_{\theta}(\mathbf{x}') \|^2] - 2\mathbb{E}_{\mathbf{x},\mathbf{x}'}\Big[\Big\langle \nabla_{\mathbf{x}'}\log{q(\mathbf{x}'|\mathbf{x})},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\Big] + \underbrace{\mathbb{E}_{\mathbf{x},\mathbf{x}'}[ \| \nabla_{\mathbf{x}'}\log q(\mathbf{x}'|\mathbf{x})\|^2]}_{\text{independent of \ $\theta$}} \\ 
&= \mathbb{E}_{\mathbf{x}'}\mathbb{E}_{\mathbf{x}|\mathbf{x}'}[\|\mathbf{s}_{\theta}(\mathbf{x}') \|^2] - 2\mathbb{E}_{\mathbf{x},\mathbf{x}'}\Big[\Big\langle \nabla_{\mathbf{x}'}\log{q(\mathbf{x}'|\mathbf{x})},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\Big] + \underbrace{\mathbb{E}_{\mathbf{x},\mathbf{x}'}[ \| \nabla_{\mathbf{x}'}\log q(\mathbf{x}'|\mathbf{x})\|^2]}_{\text{independent of \ $\theta$}}\\ 
&= \mathbb{E}_{\mathbf{x}'}[\|\mathbf{s}_{\theta}(\mathbf{x}') \|^2] - 2\mathbb{E}_{\mathbf{x},\mathbf{x}'}\Big[\Big\langle \nabla_{\mathbf{x}'}\log{q(\mathbf{x}'|\mathbf{x})},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\Big] + \underbrace{\mathbb{E}_{\mathbf{x},\mathbf{x}'}[ \| \nabla_{\mathbf{x}'}\log q(\mathbf{x}'|\mathbf{x})\|^2]}_{\text{independent of \ $\theta$}}
\end{aligned}
$$

Looking at the two objectives, the first two terms match and the third terms are constant with respect to $\theta$. So we conclude that $\argmin_{\theta} \bar{J}(\theta) = \argmin_{\theta} J(\theta)$. For learning, we need the gradient of this objective, 

$$
\begin{aligned}
\nabla_{\theta}\bar{J}(\theta)  &= \nabla_{\theta}\mathbb{E}_{\mathbf{x},\mathbf{x}'}[ \| \nabla_{\mathbf{x}'}\log q(\mathbf{x}'|\mathbf{x}) - \mathbf{s}_{\theta}(\mathbf{x}') \|^2] \\ 
&=  \mathbb{E}_{\mathbf{x},\mathbf{x}'}[ \nabla_{\theta}\| \nabla_{\mathbf{x}'}\log q(\mathbf{x}'|\mathbf{x}) - \mathbf{s}_{\theta}(\mathbf{x}') \|^2] \\ 
&= 2\mathbb{E}_{\mathbf{x},\mathbf{x}'}\Big[\Big(\nabla_{\theta}\mathbf{s}_{\theta}(\mathbf{x}')\Big)\Big(\mathbf{s}_{\theta}(\mathbf{x}') - \nabla_{\mathbf{x}'}\log q(\mathbf{x}'|\mathbf{x})\Big)\Big]
\end{aligned}
$$

Where we assume the score function approximator is differentiable and has jacobian,

$$
\nabla_{\theta}\mathbf{s}_{\theta}(\mathbf{x}') = \begin{bmatrix}
    - & \frac{d\mathbf{s}_{\theta}}{d\theta_1}^T & -  \\\ 
    
    - & \frac{d\mathbf{s}_{\theta}}{d\theta_2}^T & -  \\\ 
    - &\vdots & - \\\ 
    - & \frac{d\mathbf{s}_{\theta}}{d\theta_n}^T & -
\end{bmatrix}
$$

## Review

So, when it comes to score matching here is progression of objectives we have derived and their reasoning. 

#### 1. Score Matching

The canonical score matching objective is,

$$
\min_\theta \quad J_{1}(\theta) := \mathbb{E}_{\hat{p}(\mathbf{x})}[ \| \nabla_{\mathbf{x}}\log \hat{p}(\mathbf{x}) - \mathbf{s}_{\theta}(\mathbf{x})  \|^2]
$$

This is difficult in practice because we do not have access to the target score function $\nabla_{\mathbf{x}}\log{\hat{p}(\mathbf{x})}$. To overcome this difficulty, we have implicit score matching. 

$$
\min_{\theta}\quad J_{2}(\theta) := \mathbb{E}_{\hat{p}(\mathbf{x})}[\|\mathbf{s}_{\theta}(\mathbf{x})\|^2 + 2\ \text{tr}\big(\nabla_{\mathbf{x}}\mathbf{s}_{\theta}(\mathbf{x}\big))]
$$

We showed that $\argmin_{\theta} J_{ISM}(\theta) = \argmin_{\theta} J_{ESM}(\theta)$, so we can use the implicit score matching objective minimize the original explicit score matching objective. The difficulty here is that for high-dimensional data, materializing the jacobian $\nabla_{\mathbf{x}}\mathbf{s}_{\theta}(\mathbf{x})$ is not efficient. For example, when the dimension of the data is $d = 1000$, we would require $d$ backward passes to materialize each derivative required to compute the trace.

#### 2. Denoising Score Matching
To overcome this difficulty, we perturb our data distribution with some noise to obtain a slightly noisy distribution over our data, $q(\mathbf{x}') = \int p(\mathbf{x}) q(\mathbf{x}' | \mathbf{x}) \ d\mathbf{x}$. The idea is that when the noise is small we can say that the score functions of the perturbed and true data distributions will be approximately the same. Starting with the canonical score matching objective for the perturbed distribution,

$$
\min_{\theta} \quad J(\theta) := \mathbb{E}_{\mathbf{x}'}[ \| \nabla_{\mathbf{x}'}\log q(\mathbf{x}') - \mathbf{s}_{\theta}(\mathbf{x}') \|^2]
$$

In practice this is difficult to optimize because we do not have access to the score function of the perturbed distribution, $\nabla_{\mathbf{x}'}\log q(\mathbf{x}')$. However, we can optimize another objective $\bar{J}(\theta)$ and showed that 
$\argmin_{\theta} J(\theta) = \argmin_{\theta} \bar{J}(\theta)$.

$$
\min_{\theta} \quad \bar{J}(\theta) := \mathbb{E}_{\mathbf{x},\mathbf{x}'}[ \| \nabla_{\mathbf{x}'}\log q(\mathbf{x}'|\mathbf{x}) - \mathbf{s}_{\theta}(\mathbf{x}') \|^2]
$$

# Noise Conditional Score Networks
Suppose we have data that follows a probability distribution $ \mathbf{x} \sim p(\mathbf{x})$. We know that we can perturb the data with small noise to get a perturbed distribution $\mathbf{x}' \sim q(\mathbf{x}') = \int p(\mathbf{x}) q(\mathbf{x}' | \mathbf{x})\ d\mathbf{x}$ and then train a score function $\mathbf{s}\_{\theta}(\mathbf{x}')$ so that is approximates the score of the noisy distribution well, $ \mathbf{s}\_{\theta}(\mathbf{x}') \approx \nabla\_{\mathbf{x}'}\log{q}(\mathbf{x}')$. We can do this by minimizing the following objective: 

$$
\min_{\theta} \quad \bar{J}(\theta) := \mathbb{E}_{\mathbf{x},\mathbf{x}'}[ \| \nabla_{\mathbf{x}'}\log q(\mathbf{x}'|\mathbf{x}) - \mathbf{s}_{\theta}(\mathbf{x}') \|^2]
$$

There are two main problems that are addressed by this approach:

1. The density $p(\mathbf{x})$ is likely to be close to zero for large volumes in $\mathbb{R}^d$. In other words, most of the density will be concentrated in pockets of $\mathbb{R}^d$. As a result, there will not be a strong enough score signal to train our score network in low density regions, and its estimate in those regions will often be inaccurate. This can lead to problems with langevin dynamics if we begin with a sample from a low density region. By adding large amounts of noise to the data distribution, we are filling the regions of low density with a signal that we can then train on.

2. Manifold hypothesis. This states that most of the data tends to concentrate on low dimensional manifolds in high dimensional space. Since the score function is a gradient taken on the whole space, it is undefined if $\mathbf{x}$ resides on a low dimensional manifold. By adding noise to the data, the support of the perturbed distribution is all of $\mathbb{R}^d$, (i.e. the perturbed data is not confined to a low dimensional manifold).

### References

Vincent, Pascal. "A connection between score matching and denoising autoencoders." Neural computation 23.7 (2011): 1661-1674.
