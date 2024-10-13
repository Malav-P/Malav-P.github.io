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

## Learning Energy Based Models
In generative modeling, we often have i.i.d samples of data from a random variable $\mathbf{x} \sim \hat{p}(\mathbf{x})$. However, we do not have the form of $\hat{p}(\mathbf{x})$ and so cannot generate more samples. So, we create a model of the distribution parameterized by $\theta$ of the form:

$$
p(\mathbf{x}) := \frac{e^{-f_{\theta}(\mathbf{x})}}{Z(\theta)}
$$

where $f_{\theta}(\mathbf{x})$ is a function differentiable in $\theta$, often a neural network, and $Z(\theta) := \int e^{-f_{\theta}(\mathbf{x})}\ d\mathbf{x}$ is the normalizing constant. Given a dataset $\mathcal{D} = \{\mathbf{x}\_i \}\_{i=1}^{n}$, we choose $\theta$ by maximizing the usual log-likelihood,

$$
\max_{\theta} \quad l(\theta) := - \log{Z(\theta)} - \frac{1}{n}\sum_i f_{\theta}(\mathbf{x}_i) 
$$

Taking the gradient, 

$$
\begin{aligned}
\nabla_{\theta}\ l(\theta) &= -\nabla_{\theta}\log{Z(\theta)} - \frac{1}{n}\sum_i \nabla_{\theta}f_{\theta}(\mathbf{x}_i) \\ 
&= \mathbb{E}_{p(\mathbf{x})}[\nabla_{\theta}f_{\theta}(\mathbf{x})] - \frac{1}{n}\sum_i \nabla_{\theta}f_{\theta}(\mathbf{x}_i)
\end{aligned}
$$

In theory, we can use Markov-Chain Monte Carlo (MCMC) methods estimate the expectation with an average over many samples. 
There are two main problems with this approach. 

1. MCMC is usually very slow as it takes many steps to run the chain until equilibrium and begin generating samples from $p(\mathbf{x})$. 
2. Calculating $Z(\theta)$ is usually intractable as its an integral over all space. As a result, we cannot calculate $p(\mathbf{x})$ for a given $\mathbf{x}$.

Score matching attempts to remedy problem (1) by doing away with the often intractable $Z(\theta)$.

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
One problem with implicit score matching is that materializing the jacobian $\nabla\_{\mathbf{x}}\mathbf{s}\_{\theta}(\mathbf{x})$ becomes difficult. For example, when the dimension of the data is $d = 1000$, we would require $d$ backward passes to materialize each derivative required to compute the trace. To overcome this let us introduce another model $q(\mathbf{x}'|\mathbf{x})$ that perturbs the data with noise. Then $q(\mathbf{x}') = \int p(\mathbf{x})q(\mathbf{x}'|\mathbf{x})\ d\mathbf{x} $. The intuition is that if the noise is small enough, then the score function of the perturbed distribution will be approximately equal to that of the true distribution. Then we can run Langevin dynamics on this perturbed distribution. Our objective is

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

We showed that $\argmin_{\theta} J_{1}(\theta) = \argmin_{\theta} J_{2}(\theta)$, so we can use the implicit score matching objective minimize the original explicit score matching objective. The difficulty here is that for high-dimensional data, materializing the jacobian $\nabla_{\mathbf{x}}\mathbf{s}_{\theta}(\mathbf{x})$ is not efficient. 

#### 2. Denoising Score Matching
To overcome this difficulty, we perturb our data distribution with some noise to obtain a slightly noisy distribution over our data, $q(\mathbf{x}') = \int p(\mathbf{x}) q(\mathbf{x}' | \mathbf{x}) \ d\mathbf{x}$. The idea is that when the noise is small we can say that the score functions of the perturbed and true data distributions will be approximately the same. Starting with the implicit score matching objective for the perturbed distribution,

$$
\min_{\theta}\quad J(\theta) := \mathbb{E}_{\mathbf{x}'}[\|\mathbf{s}_{\theta}(\mathbf{x}')\|^2 + 2\ \text{tr}\big(\nabla_{\mathbf{x}}\mathbf{s}_{\theta}(\mathbf{x}')\big)]
$$

In practice this is difficult to optimize because we cannot sample $\mathbf{x}' \sim q(\mathbf{x}')$. However, we can optimize another objective $\bar{J}(\theta)$ and showed that 
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

There are still 2 problems with denoising score matching. 

1. There is a tradeoff between adding noise to fill low density regions and corrupting the original data distribution. Too much noise added and the score function is no longer a good approximator of the score function of the data distribution. 
2. If there are two modes of $p(\mathbf{x})$ connected by low density regions, langevin dynamics will struggle to capture the relative weights of the two modes. We may need very small step sizes and a large number of steps to get accurate samples. Furthermore, if we start near one mode, it may take a large number of steps for langevin dynamics to begin sampling from the second mode.

To solve the first problem, what if we progressively add noise to the data distribution with a noise schedule $\sigma_1 > \sigma_2 > \ldots > \sigma_L$ and then train a score network on all the noise scales, i.e. $\mathbf{s}_{\theta}(\mathbf{x}, \sigma)$?. Then, when sampling, we use initially use scores corresponding to large noise and gradually turn the noise down as we reach areas of larger density. This method optimally leverages the tradeoff described in problem 1. This is the essence of Noise Conditional Score Networks.

### Learning NCSNs
We follow the the work of [2] in outlining NCSN's. Define a set of noise scales $\{\sigma\_i\}\_{i=1}^{L}$ that follows a geometric sequence: $\frac{\sigma_1}{\sigma_2} = \cdots = \frac{\sigma_{L-1}}{\sigma_L} > 1 $. Define the noisy distribution and the conditional distribution:

$$
\begin{aligned}
q_{\sigma}(\mathbf{x}') :&= \int p(\mathbf{x}) \ \underbrace{\mathcal{N}(\mathbf{x}, \sigma^2\mathbf{I})}_{q(\mathbf{x}' | \mathbf{x})}\ d\mathbf{x} \\ 
\end{aligned}
$$

We train the score network such that $\mathbf{s}\_{\theta}(\mathbf{x}', \sigma) \approx \nabla\_{\mathbf{x}'}\log{q\_{\sigma}(\mathbf{x}')}$.

The objective for a particular value of $\sigma$ follows exactly from denoising score matching:

$$
\begin{aligned}
\min_{\theta} \quad l(\theta, \sigma) :&= \mathbb{E}_{\mathbf{x},\mathbf{x}'}[ \| \nabla_{\mathbf{x}'}\log q(\mathbf{x}'|\mathbf{x}) - \mathbf{s}_{\theta}(\mathbf{x}', \sigma) \|^2] \\ 
&= \mathbb{E}_{\mathbf{x},\mathbf{x}'}[ \| \frac{\mathbf{x}' - \mathbf{x}}{\sigma^2} + \mathbf{s}_{\theta}(\mathbf{x}', \sigma) \|^2]
\end{aligned}
$$

Define the total loss objective over all values $\sigma_i$ as 

$$
\min_{\theta} \quad l(\theta) := \sum_i \lambda(\sigma_i)\ l(\theta, \sigma_i)
$$

where $\lambda(\sigma)$ is a weighting constant that ensures each term is approximately the same order of magnitude. This ensures that no particular value of $\sigma$ is overemphasizes in the minimization objective. The authors of [2] find that empirically, $\lambda(\sigma) = \sigma^2$ is a good choice.

### Inference with NCSNs
The intuition around sampling is as follows. 

1. Suppose we sample an intial point $\mathbf{x}_0 \sim \pi(\mathbf{x}_0)$. It is likely to be in a low density region of the data distribution $p(\mathbf{x})$.
2. To get a good score signal, we use the score network with the largest value of $\sigma$ and begin sampling via langevin dynamics. Sample for some number of iterations.

3. By this point, we have likely begun to move closer to regions of high density. So, we reduce the noise $\sigma$ and begin to take smaller step sizes as we sample with langevin dynamics.

4. Repeat step 3 until we have reached sufficiently small $\sigma$ such that samples now are $\sim p(\mathbf{x})$.


The authors of [2] present the algorithm, called annealed langevin dynamics.

## Toward Stochastic Differential Equations

Let $\mathbf{x} \sim p(\mathbf{x})$ denote data sampled from the data distribution. Let $q\_{\sigma}(\mathbf{x}' \| \mathbf{x}):= \mathcal{N}(\mathbf{x}, \sigma^2\mathbf{I})$ be a perturbation kernel at $q\_{\sigma}(\mathbf{x}') := \int p(\mathbf{x})\  q\_{\sigma}(\mathbf{x}' | \mathbf{x})\ d\mathbf{x}$ be the associated perturbed distribution. Noise conditional score networks introduce a set of variances $\{ \sigma\_i\}_{i=1}^{L}$  with $\sigma_1 < \sigma_2 < \cdots < \sigma\_L$ and train a score network $\mathbf{s}\_{\theta}(\mathbf{x}', \sigma)$ such that $\mathbf{s}\_{\theta}(\mathbf{x}', \sigma) \approx \nabla\_{\mathbf{x}}\log q\_{\sigma}(\mathbf{x}')$ for all $\sigma \in \{ \sigma\_i\}_{i=1}^{L}$. Then we use langevin dynamics with this score network. In regions of low density we use larger values of sigma for a stronger score signal and tune it down as we move towards regions of high density. Now let $\mathbf{x}\_i$  be the random variable that is sampled from the perturbed distribution with parameter $\sigma\_i$, i.e. $\mathbf{x}\_i \sim q\_{\sigma\_i}(\mathbf{x}\_i)$. Note that these random variables follow a Markov chain:

$$
\mathbf{x}_{i} = \mathbf{x}_{i-1} + \sqrt{\sigma_{i}^2 - \sigma_{i-1}^2}\mathbf{z}_{i-1}, \quad \quad \mathbf{z}_{i-1} \sim \mathcal{N}(0, \mathbf{I})
$$

To see this, simply apply the formula recursively by replacing  $\mathbf{x}\_{i-1}$ above to arrive at

$$
\begin{aligned}
\mathbf{x}_i &= \mathbf{x}_0 + \sqrt{\sum_{j=1}^{i} \sigma_j^2 - \sigma_{j-1}^2}\  \mathbf{z}, \quad \quad \mathbf{z} \sim \mathcal{N}(0, \mathbf{I}) \\ 
&= \mathbf{x}_0 + \sigma_i \mathbf{z}
\end{aligned}
$$

which is consistent with the fact that $\mathbf{x}\_i \| \mathbf{x}\_0 \sim \mathcal{N}(\mathbf{x}\_0, \sigma_i^2\mathbf{I})$. Note that above we use a slight change of notation $\mathbf{x}\_0 \sim p(\mathbf{x}\_0)$ which corresponds to the true data distribution and $\sigma\_0 = 0$.

Now consider the limit $L \to \infty$. The set $\{\sigma\_i \}\_{i = 1}^{L}$ becomes $\sigma(t)$ for a continuous index $t \in [0, 1]$ rather than a discrete index $i \in \{1, 2, \ldots L\}$. Let $\mathbf{x}\big(\frac{i}{L}\big)  = \mathbf{x}\_i$ be a new way to write $\mathbf{x}\_i$,  $\sigma(\frac{i}{L}) = \sigma\_i$ be the new way to write $\sigma\_i$, and $\mathbf{z}(\frac{i}{L}) = \mathbf{z}\_i$ be the new way to write $\mathbf{z}\_i$ using the continuous index $t$. Let $\Delta t = \frac{1}{L}$. Then returning to the Markov chain we can write

$$
\begin{aligned}
\mathbf{x}(t + \Delta t) &= \mathbf{x}(t) + \sqrt{\sigma^2(t + \Delta t) - \sigma(t)}\ \mathbf{z}(t) \\ 
&\approx \mathbf{x}(t) + \sqrt{\frac{\text{d}[\sigma^2(t)]}{\text{d}t}\Delta t}\ \mathbf{z}(t), \quad \quad \text{for } \Delta t \ll 1
\end{aligned}
$$

Rewriting this, 

$$
\mathbf{x}(t + \Delta t) - \mathbf{x}(t) = \sqrt{\frac{\text{d}[\sigma^2(t)]}{\text{d}t}}\underbrace{\sqrt{\Delta t}\ \mathbf{z}(t)}_{\mathbf{w}(t + \Delta t) - \mathbf{w}(t)}
$$

Where $\mathbf{w}(t)$ is a Wiener process. To see this recall that a wiener process $\mathbf{w}(t)$ has Gaussian increments, i.e. $\mathbf{w}(t + u) - \mathbf{w}(t) \sim \mathcal{N}(0, u\mathbf{I})$. Now in the limit $\Delta t \to 0$, the above equation becomes a stochastic differential equation:

$$
\text{d}\mathbf{x} = \sqrt{\frac{\text{d}[\sigma^2(t)]}{\text{d}t}}\ \text{d}\mathbf{w}
$$

The above stochastic differential equation transforms the initial random variable $\mathbf{x}(0)$ with distribution $p$ at $t = 0$ to another random variable $\mathbf{x}(1)$ with distribution $q$ at $t=1$. This equation represents the forward diffusion process. In our case $p$ is the data distribution and $q$ is essentially a Gaussian distribution. This is because (TODO DERIVE sigma(t))

<!-- For notational convenience let $\mathbf{x}_0 \sim p(\mathbf{x}_0) $ refer to the true data distribution and $q_{\sigma}(\mathbf{x}) := \int p(\mathbf{x}_0)\ \mathcal{N}(\mathbf{x}_0, \sigma^2\mathbf{I})\ d\mathbf{x}_0$ be a perturbed distribution. Noise conditional score networks introduce a set of variances $\{\sigma_i \}_{i=1}^{L}$ and a perturbation kernel $q(\mathbf{x}_i | \mathbf{x}_0) := \mathcal{N}(\mathbf{x}_0, \sigma_i^2\mathbf{I})$ such that $\mathbf{x}_i \sim q(\mathbf{x}_i) = \int p(\mathbf{x}_0) q(\mathbf{x}_i | \mathbf{x}_0) \ d\mathbf{x}_0 $. Then we proceed to learn a score network $\mathbf{s}_{\theta}(\mathbf{x}, \sigma)$ from these perturbed distributions such that $\mathbf{s}_{\theta}(\mathbf{x}, \sigma) \approx \nabla_{\mathbf{x}}\log q_{\sigma}(\mathbf{x})$ -->

### References

1. Vincent, Pascal. "A connection between score matching and denoising autoencoders." Neural computation 23.7 (2011): 1661-1674.

2. Song, Yang, and Stefano Ermon. "Generative modeling by estimating gradients of the data distribution." Advances in neural information processing systems 32 (2019).
