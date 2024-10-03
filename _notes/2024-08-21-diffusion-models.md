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

## Diffusion Models

Define a joint distribution $q(\mathbf{x}\_0, \ldots,\mathbf{x}\_T)$ over our data variable $\mathbf{x}\_0$ and latent variables $\{\mathbf{x}\_i\}\_{i=1}^T$. We can decompose this joint distribution into a product of conditional distribution that represent a diffusion process of $\mathbf{x}\_0$ into Gaussian noise. This is called the forward process

#### Forward Process 

$$
q(\mathbf{x}_0, \ldots,\mathbf{x}_T) := q(\mathbf{x}_0)\prod_{t=1}^T q(\mathbf{x}_t | \mathbf{x}_{t-1}) \quad \quad q(\mathbf{x}_t | \mathbf{x}_{t-1}) := \mathcal{N}(\mathbf{f}_{\mathbf{\mu}}(\mathbf{x}_{t-1}),\  \mathbf{f}_{\mathbf{\Sigma}}(\mathbf{x}_{t-1}))
$$

In the DDPM paper [1], the authors choose a set of scalars $\{\beta_t\}_{t=1}^T$ and set the mean and variance of the conditional distributions as

$$
\begin{aligned}
\mathbf{f}_{\mathbf{\mu}}(\mathbf{x}_{t-1}) &:= \sqrt{1-\beta_t}\ \mathbf{x}_{t-1} \\ 
\mathbf{f}_{\mathbf{\Sigma}}(\mathbf{x}_{t-1}) &:= \beta_t\mathbf{I}
\end{aligned}
$$

Note that with this parameterization, with sufficiently large $T$, the samples $q(\mathbf {x}_T) \approx \mathcal{N}(0, \mathbf{I})$. This is easy to sample from and will be useful for the reverse process.

#### Reverse Process
An alternative way to write the joint distribution is via the reverse process, i.e. the generative process. When written this way, we sample from the joint distribution by sampling from the conditional distributions $q(\mathbf{x}_{t-1} | \mathbf{x}_t)$, starting with $q(\mathbf{x}_T)$.

$$
q(\mathbf{x}_0, \ldots,\mathbf{x}_T) := q(\mathbf{x}_T)\prod_{t=1}^T q(\mathbf{x}_{t-1} | \mathbf{x}_{t})
$$

We know that $q(\mathbf{x}\_T) \approx \mathcal{N}(0, \mathbf{I})$. Futhermore a remarkable result [2] shows that $q(\mathbf{x}\_{t-1} | \mathbf{x}\_{t})$ is also normally distributed but we do not know its mean and variance. Let us create a model for this posterior that estimates the mean and variance as functions of learnable parameters $\theta$:

$$
p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_t) := \mathcal{N}(\boldsymbol{\mu}_{\theta}(\mathbf{x}_t, t), \boldsymbol{\Sigma}_{\theta}(\mathbf{x}_t, t))
$$

And since $q(\mathbf{x}\_T) \approx \mathcal{N}(0, \mathbf{I})$, set $p(\mathbf{x}\_T) := \mathcal{N}(0, \mathbf{I})$. Then our model for the joint distribution is 

$$
p_{\theta}(\mathbf{x}_0, \ldots,\mathbf{x}_T) := p(\mathbf{x}_T)\prod_{t=1}^T p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_{t})
$$

What we usually care about is the marginal density $p_{\theta}(\mathbf{x}_0)$ which can be written as:

$$
\begin{aligned}
  p_{\theta}(\mathbf{x}_0) &= \int p_{\theta}(\mathbf{x}_0, \ldots,\mathbf{x}_T)\  d\mathbf{x}_1 \ldots d\mathbf{x}_T \\ 
  &= \int p(\mathbf{x}_T)\prod_{t=1}^T p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_{t})\  d\mathbf{x}_1 \ldots d\mathbf{x}_T
\end{aligned}
$$

Now this integral is naively intractable. But we can rewrite it as,

$$
\begin{aligned}
  p_{\theta}(\mathbf{x}_0)
  &= \int p(\mathbf{x}_T)\prod_{t=1}^T p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_{t})\  d\mathbf{x}_1 \ldots d\mathbf{x}_T \\ 
  &= \int p(\mathbf{x}_T)\ q(\mathbf{x}_1, \ldots, \mathbf{x}_T | \mathbf{x}_0) \frac{\prod_{t=1}^T p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_{t})}{q(\mathbf{x}_1, \ldots, \mathbf{x}_T | \mathbf{x}_0)}\  d\mathbf{x}_1 \ldots d\mathbf{x}_T \\ 
  &= \int q(\mathbf{x}_1, \ldots, \mathbf{x}_T | \mathbf{x}_0)\ p(\mathbf{x}_T) \prod_{t=1}^{T}\frac{p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_{t})}{q(\mathbf{x}_t | \mathbf{x}_{t-1})}\  d\mathbf{x}_1 \ldots d\mathbf{x}_T \\ 
  &= \mathbb{E}_{q(\mathbf{x}_1, \ldots, \mathbf{x}_T | \mathbf{x}_0)}\Bigg[p(\mathbf{x}_T) \prod_{t=1}^{T}\frac{p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_{t})}{q(\mathbf{x}_t | \mathbf{x}_{t-1})}\Bigg]
\end{aligned}
$$

This is an expectation over forward trajectories conditioned on $\mathbf{x}_0$, so we can estimate the model probability by rapidly averaging over many forward trajectories.

### Learning the Model
To learn the model we maximize its log-likelihood over the data distribution:

$$
\begin{aligned}
\max_{\theta} \quad \ell(\theta) :&= \mathbb{E}_{q(\mathbf{x}_0)}[\log{p_{\theta}(\mathbf{x}_0)}] \\ 
&= \mathbb{E}_{q(\mathbf{x}_0)}\Bigg[ \log{\bigg(\mathbb{E}_{q(\mathbf{x}_1, \ldots, \mathbf{x}_T | \mathbf{x}_0)}\Bigg[p(\mathbf{x}_T) \prod_{t=1}^{T}\frac{p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_{t})}{q(\mathbf{x}_t | \mathbf{x}_{t-1})}\Bigg]\bigg)} \Bigg] \\ 
&\geq \mathbb{E}_{q(\mathbf{x}_0)}\mathbb{E}_{q(\mathbf{x}_1, \ldots, \mathbf{x}_T | \mathbf{x}_0)}\Bigg[ \log{\bigg(p(\mathbf{x}_T) \prod_{t=1}^{T}\frac{p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_{t})}{q(\mathbf{x}_t | \mathbf{x}_{t-1})}\bigg)} \Bigg] \quad \quad (\text{Jensen's inequality}) \\ 
&= \mathbb{E}_{q(\mathbf{x}_0)}\mathbb{E}_{q(\mathbf{x}_1, \ldots, \mathbf{x}_T | \mathbf{x}_0)}\Bigg[\log{p(\mathbf{x}_T)} + \sum_{t=1}^{T} \log{\frac{p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_{t})}{q(\mathbf{x}_t | \mathbf{x}_{t-1})}}\Bigg] \\ 
&= \mathbb{E}_{q(\mathbf{x}_0)}\mathbb{E}_{q(\mathbf{x}_1, \ldots, \mathbf{x}_T | \mathbf{x}_0)}\Bigg[\log{p(\mathbf{x}_T)} + \log{\frac{p_{\theta}(\mathbf{x}_{0} | \mathbf{x}_{1})}{q(\mathbf{x}_1 | \mathbf{x}_{0})}} +\sum_{t=2}^{T} \log{\frac{p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_{t})}{q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0)}}\Bigg] \quad \quad (\text{Markov property implies}\ \  q(\mathbf{x}_t | \mathbf{x}_{t-1}) = q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0)) \\ 
&= \mathbb{E}_{q(\mathbf{x}_0)}\mathbb{E}_{q(\mathbf{x}_1, \ldots, \mathbf{x}_T | \mathbf{x}_0)}\Bigg[\log{p(\mathbf{x}_T)} + \log{\frac{p_{\theta}(\mathbf{x}_{0} | \mathbf{x}_{1})}{q(\mathbf{x}_1 | \mathbf{x}_{0})}} +\sum_{t=2}^{T} \log{\frac{p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_{t})}{q(\mathbf{x}_{t-1} | \mathbf{x}_{t}, \mathbf{x}_0)}\cdot\frac{q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t | \mathbf{x}_0)}}\Bigg] \quad \quad (\text{Bayes Rule}) \\ 
&= \mathbb{E}_{q(\mathbf{x}_0)}\mathbb{E}_{q(\mathbf{x}_1, \ldots, \mathbf{x}_T | \mathbf{x}_0)}\Bigg[\log{\frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T | \mathbf{x}_0)}} + \log{p_{\theta}(\mathbf{x}_{0} | \mathbf{x}_{1})} +\sum_{t=2}^{T} \log{\frac{p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_{t})}{q(\mathbf{x}_{t-1} | \mathbf{x}_{t}, \mathbf{x}_0)}}\Bigg]\\ 
&=\mathbb{E}_{q(\mathbf{x}_0)}\Bigg[\mathbb{E}_{q(\mathbf{x}_1, \ldots, \mathbf{x}_T | \mathbf{x}_0)}\bigg[\log{\frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T | \mathbf{x}_0)}}\bigg] + \mathbb{E}_{q(\mathbf{x}_1, \ldots, \mathbf{x}_T | \mathbf{x}_0)}\Big[\log{p_{\theta}(\mathbf{x}_{0} | \mathbf{x}_{1})}\Big] +\sum_{t=2}^{T} \mathbb{E}_{q(\mathbf{x}_1, \ldots, \mathbf{x}_T | \mathbf{x}_0)}\bigg[\log{\frac{p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_{t})}{q(\mathbf{x}_{t-1} | \mathbf{x}_{t}, \mathbf{x}_0)}}\bigg]\Bigg] \\ 
&=\mathbb{E}_{q(\mathbf{x}_0)}\Bigg[\mathbb{E}_{q( \mathbf{x}_T | \mathbf{x}_0)}\bigg[\log{\frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T | \mathbf{x}_0)}}\bigg] + \mathbb{E}_{q(\mathbf{x}_1 | \mathbf{x}_0)}\Big[\log{p_{\theta}(\mathbf{x}_{0} | \mathbf{x}_{1})}\Big] +\sum_{t=2}^{T} \mathbb{E}_{q(\mathbf{x}_{t-1},\mathbf{x}_t | \mathbf{x}_0)}\bigg[\log{\frac{p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_{t})}{q(\mathbf{x}_{t-1} | \mathbf{x}_{t}, \mathbf{x}_0)}}\bigg]\Bigg] \quad \quad \text{Marginalize}\ \ q \\ 
&=\mathbb{E}_{q(\mathbf{x}_0)}\Bigg[-\text{D}_{\text{KL}}\Big(q(\mathbf{x}_T|\mathbf{x}_0)\  \| \ p(\mathbf{x}_T) \Big) + \mathbb{E}_{q(\mathbf{x}_1 | \mathbf{x}_0)}\Big[\log{p_{\theta}(\mathbf{x}_{0} | \mathbf{x}_{1})}\Big] +\sum_{t=2}^{T} \mathbb{E}_{q(\mathbf{x}_t | \mathbf{x}_0)} \mathbb{E}_{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)}\bigg[\log{\frac{p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_{t})}{q(\mathbf{x}_{t-1} | \mathbf{x}_{t}, \mathbf{x}_0)}}\bigg]\Bigg] \\ 
&=\mathbb{E}_{q(\mathbf{x}_0)}\Bigg[-\text{D}_{\text{KL}}\Big(q(\mathbf{x}_T|\mathbf{x}_0)\  \| \ p(\mathbf{x}_T) \Big) + \mathbb{E}_{q(\mathbf{x}_1 | \mathbf{x}_0)}\Big[\log{p_{\theta}(\mathbf{x}_{0} | \mathbf{x}_{1})}\Big] - \sum_{t=2}^{T} \mathbb{E}_{q(\mathbf{x}_t | \mathbf{x}_0)}\Big[ \text{D}_{\text{KL}}\Big( q(\mathbf{x}_{t-1} | \mathbf{x}_{t}, \mathbf{x}_0) \ \| \ p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_{t})\Big)\Big] \Bigg]
\end{aligned}
$$

In the last line, we have two distributions that we know the form of: $q(\mathbf{x}_t | \mathbf{x}_0)$ and $q(\mathbf{x}_{t-1}, | \mathbf{x}_t, \mathbf{x}_0)$. We show below that they are Gaussian distributions!

#### Forward process posterior $q(\mathbf{x}_t | \mathbf{x}_0)$

We use the reparameterization trick. It is well known that an affine transformation of a gaussian random variable is another gaussian random variable. Define $\alpha_t := 1-\beta_t $ Since $q(\mathbf{x}_t | \mathbf{x}_{t-1}) := \mathcal{N}(\sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$, we can write $\mathbf{x}_t$ as,

$$
\begin{aligned}
\mathbf{x}_t &= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_t  &&\boldsymbol{\epsilon}_t \sim \mathcal{N}(0, \mathbf{I}) \\ 
&= \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{1-\alpha_{t-1}}\boldsymbol{\epsilon}_{t-1}) + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_t \\ 
&\ \ \vdots \\ 
&= \sqrt{\alpha_t\alpha_{t-1}\ldots \alpha_1}\mathbf{x}_0 + \sum_{i=1}^{t} \sqrt{\alpha_{t}\alpha_{t-1}\ldots \alpha_{i+1}}\sqrt{1-\alpha_i}\boldsymbol{\epsilon}_i \\ 
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha_t}}\boldsymbol{\epsilon}  &&\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}),\ \  \bar{\alpha_t} := \prod_{i=1}^t \alpha_i
\end{aligned}
$$

We outline the justification from the second last to last line. We assume that $\{\boldsymbol{\epsilon}\_i\}$ are a set of i.i.d. standard normal random variables. We also know that for two independent random vectors $\mathbf{u}\_1 \sim \mathcal{N}(0, \sigma\_1^2\mathbf{I})$ and $\mathbf{u}\_2 \sim \mathcal{N}(0, \sigma\_2^2\mathbf{I})$, their sum also follows a normal distribution, $\mathbf{u}\_1 + \mathbf{u}\_2 \sim \mathcal{N}(0, (\sigma\_1^2 + \sigma\_2^2)\mathbf{I}) $. Using this fact, we can derive the distribution of the sum above,

$$
\begin{aligned}
\mathbf{z}:&=\sum_{i=1}^{t} \sqrt{\alpha_{t}\alpha_{t-1}\ldots \alpha_{i+1}}\sqrt{1-\alpha_i}\boldsymbol{\epsilon}_i \quad \sim \quad \mathcal{N}(0, \sigma^2\mathbf{I}) \\ 
\sigma &= \sqrt{\sum_{i=1}^t(\alpha_{t}\alpha_{t-1}\ldots \alpha_{i+1})(1-\alpha_i)} \\ 
&= \sqrt{(\alpha_{t}\alpha_{t-1}\ldots \alpha_{2})(1-\alpha_1) + (\alpha_{t}\alpha_{t-1}\ldots \alpha_{3})(1-\alpha_2)+\ldots+(1-\alpha_{t-1})\alpha_t + (1-\alpha_t)} \\ 
&= \sqrt{-\alpha_t\alpha_{t-1}\ldots\alpha_1 + (\alpha_t\alpha_{t-1}\ldots\alpha_2 -\alpha_t\alpha_{t-1}\ldots\alpha_2 ) + \ldots +(\alpha_t - \alpha_t) + 1} \\ 
&= \sqrt{1 - \alpha_1\alpha_2\ldots\alpha_{t}} \\ 
&= \sqrt{1 - \bar{\alpha_t}}
\end{aligned}
$$

So we may write $ \mathbf{z} = \sqrt{1-\bar{\alpha}\_t}\boldsymbol{\epsilon}$ with $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}) $. Therefore, $q(\mathbf{x}\_t|\mathbf{x}\_0) = \mathcal{N}(\sqrt{\bar{\alpha\_t}}\mathbf{x}\_0, (1-\bar{\alpha\_t})\mathbf{I})$. 