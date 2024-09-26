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

In this note, we describe score matching.

Consider a dataset $\mathcal{D} = \{\mathbf{x}_i\}\\_{i=1}^{n}$ where $\mathbf{x} \in \mathbb{R}^d$. Consider an energy based model

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
$$

## Score Matching
Suppose we do not wish to work with the intractable $Z(\theta)$. Consider then the problem of finding a distribution $p$  that minimizes the Fisher divergence between our proposed distribution $p$ and the data distribution $\hat{p}$:

$$
\begin{aligned}
D_F(p \ ||\  \hat{p}) :&= \mathbb{E}_{\hat{p}(\mathbf{x})}[ \| \nabla_{\mathbf{x}}\log \hat{p}(\mathbf{x}) - \nabla_{\mathbf{x}
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

Instead of searching for the distribution $p$, we can instead search for the score function directly. More formally, we let $\mathbf{s}_{\theta}(\mathbf{x}) = \nabla_{\mathbf{x}}\log{p(\mathbf{x})}$ and optimize over $\theta$ instead:

$$
\min_{\theta}\quad  \mathbb{E}_{\hat{p}(\mathbf{x})}[\|\mathbf{s}_{\theta}(\mathbf{x})\|^2 + 2\ \text{tr}\big(\nabla_{\mathbf{x}}\mathbf{s}_{\theta}(\mathbf{x}\big))]
$$

Now with a model of the score function, we can generate samples from the data distribution using Langevin dynamics:

1. Sample $\mathbf{x}_0 \sim \pi(\mathbf{x})$ from some prior distribution $\pi$.
2. Recursively compute $\mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\epsilon}{2}\mathbf{s}_{\theta}(\mathbf{x}_{t-1}) + \sqrt{\epsilon}\ \mathbf{z}_t$, where $\mathbf{z}_t \sim \mathcal{N}(0, I)$ and some small $\epsilon > 0$.

For large number of recursive iterations, the samples $\mathbf{x}_t$ will start to converge to samples from the true distribution (under some regularity conditions, which for practical purposes are often ignored). 

# Denoising Score Matching
Suppose that materializing the jacobian $\nabla_{\mathbf{x}}\mathbf{s}_{\theta}(\mathbf{x})$ becomes difficult. Let us introduce another model $p(\mathbf{x}'|\mathbf{x})$ that perturbs the data with noise. Then $p(\mathbf{x}') = \int p(\mathbf{x})p(\mathbf{x}'|\mathbf{x})\ d\mathbf{x} $. The intuition is that if the noise is small enough, then the score function of the perturbed distribution will be approximately equal to that of the true distribution. Then we can run Langevin dynamics on this perturbed distribution. Our objective is

$$
\begin{aligned}
J(\theta) &= \mathbb{E}_{\mathbf{x}'}[ \| \nabla_{\mathbf{x}'}\log p(\mathbf{x}') - \mathbf{s}_{\theta}(\mathbf{x}') \|^2] \\ 
&= \mathbb{E}_{\mathbf{x}'}[\|\mathbf{s}_{\theta}(\mathbf{x}') \|^2] - 2\mathbb{E}_{\mathbf{x}'}[\langle \nabla_{\mathbf{x}'}\log p(\mathbf{x}'), \mathbf{s}_{\theta}(\mathbf{x}')  \rangle ] + \underbrace{\mathbb{E}_{\mathbf{x}'}[\|\nabla_{\mathbf{x}'}\log p(\mathbf{x}')\|^2]}_{\text{independent of \  $\theta$}}
\end{aligned}
$$

Now let us take a look at the second term in this sum,

$$
\begin{aligned}
\mathbb{E}_{\mathbf{x}'}[\langle \nabla_{\mathbf{x}'}\log p(\mathbf{x}'), \ \mathbf{s}_{\theta}(\mathbf{x}')  \rangle ] &= \int p(\mathbf{x}') \langle \nabla_{\mathbf{x}'}\log p(\mathbf{x}'),\  \mathbf{s}_{\theta}(\mathbf{x}')\rangle\  d\mathbf{x}' \\ &= \int p(\mathbf{x}')\Big\langle \frac{\nabla_{\mathbf{x}'}p(\mathbf{x}')}{p(\mathbf{x}')},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\ d\mathbf{x}' \\ 
&= \int \Big\langle \nabla_{\mathbf{x}'}p(\mathbf{x}'),\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\ d\mathbf{x}' \\ 
&= \int \Big\langle \nabla_{\mathbf{x}'}\int p(\mathbf{x})p(\mathbf{x}'|\mathbf{x})\ d\mathbf{x},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\ d\mathbf{x}' \\ 
&= \int \Big\langle \int p(\mathbf{x})\nabla_{\mathbf{x}'}p(\mathbf{x}'|\mathbf{x})\ d\mathbf{x},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\ d\mathbf{x}' \\ 
&= \int \Big\langle \int p(\mathbf{x})p(\mathbf{x}'|\mathbf{x})\nabla_{\mathbf{x}'}\log{p(\mathbf{x}'|\mathbf{x})}\ d\mathbf{x},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\ d\mathbf{x}' \\ 
&= \int \int p(\mathbf{x})p(\mathbf{x}'|\mathbf{x})\Big\langle \nabla_{\mathbf{x}'}\log{p(\mathbf{x}'|\mathbf{x})},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\ d\mathbf{x}'d\mathbf{x} \\ 
&= \mathbb{E}_{\mathbf{x},\mathbf{x}'}\Big[\Big\langle \nabla_{\mathbf{x}'}\log{p(\mathbf{x}'|\mathbf{x})},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\Big]
\end{aligned}
$$

So our score matching objective is 
$$
\begin{aligned}
J(\theta) &= \mathbb{E}_{\mathbf{x}'}[\|\mathbf{s}_{\theta}(\mathbf{x}') \|^2] - 2\mathbb{E}_{\mathbf{x}'}[\langle \nabla_{\mathbf{x}'}\log p(\mathbf{x}'), \mathbf{s}_{\theta}(\mathbf{x}')  \rangle ] + \underbrace{\mathbb{E}_{\mathbf{x}'}[\|\nabla_{\mathbf{x}'}\log p(\mathbf{x}')\|^2]}_{\text{independent of \  $\theta$}} \\ 
&= \mathbb{E}_{\mathbf{x}'}[\|\mathbf{s}_{\theta}(\mathbf{x}') \|^2] - 2\mathbb{E}_{\mathbf{x},\mathbf{x}'}\Big[\Big\langle \nabla_{\mathbf{x}'}\log{p(\mathbf{x}'|\mathbf{x})},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\Big] + \underbrace{\mathbb{E}_{\mathbf{x}'}[\|\nabla_{\mathbf{x}'}\log p(\mathbf{x}')\|^2]}_{\text{independent of \  $\theta$}}
\end{aligned}
$$

In practice, we cannot estimate the first term in this objective because the marginal distribution $p(\mathbf{x}')$ is unavailable. We show that another objective is equivalent (the argmin of both objectives is the same). Consider the new objective,

$$
\begin{aligned}
\bar{J}(\theta) :&= \mathbb{E}_{\mathbf{x},\mathbf{x}'}[ \| \nabla_{\mathbf{x}'}\log p(\mathbf{x}'|\mathbf{x}) - \mathbf{s}_{\theta}(\mathbf{x}') \|^2] \\ 
&= \mathbb{E}_{\mathbf{x},\mathbf{x}'}[\|\mathbf{s}_{\theta}(\mathbf{x}') \|^2] - 2\mathbb{E}_{\mathbf{x},\mathbf{x}'}\Big[\Big\langle \nabla_{\mathbf{x}'}\log{p(\mathbf{x}'|\mathbf{x})},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\Big] + \underbrace{\mathbb{E}_{\mathbf{x},\mathbf{x}'}[ \| \nabla_{\mathbf{x}'}\log p(\mathbf{x}'|\mathbf{x})\|^2]}_{\text{independent of \ $\theta$}} \\ 
&= \mathbb{E}_{\mathbf{x}'}\mathbb{E}_{\mathbf{x}|\mathbf{x}'}[\|\mathbf{s}_{\theta}(\mathbf{x}') \|^2] - 2\mathbb{E}_{\mathbf{x},\mathbf{x}'}\Big[\Big\langle \nabla_{\mathbf{x}'}\log{p(\mathbf{x}'|\mathbf{x})},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\Big] + \underbrace{\mathbb{E}_{\mathbf{x},\mathbf{x}'}[ \| \nabla_{\mathbf{x}'}\log p(\mathbf{x}'|\mathbf{x})\|^2]}_{\text{independent of \ $\theta$}}\\ 
&= \mathbb{E}_{\mathbf{x}'}[\|\mathbf{s}_{\theta}(\mathbf{x}') \|^2] - 2\mathbb{E}_{\mathbf{x},\mathbf{x}'}\Big[\Big\langle \nabla_{\mathbf{x}'}\log{p(\mathbf{x}'|\mathbf{x})},\  \mathbf{s}_{\theta}(\mathbf{x}') \Big\rangle\Big] + \underbrace{\mathbb{E}_{\mathbf{x},\mathbf{x}'}[ \| \nabla_{\mathbf{x}'}\log p(\mathbf{x}'|\mathbf{x})\|^2]}_{\text{independent of \ $\theta$}}
\end{aligned}
$$

Looking at the two objectives, the first two terms match and the third terms are constant with respect to $\theta$. So we conclude that $\argmin_{\theta} \bar{J}(\theta) = \argmin_{\theta} J(\theta)$.

### References

Vincent, Pascal. "A connection between score matching and denoising autoencoders." Neural computation 23.7 (2011): 1661-1674.
