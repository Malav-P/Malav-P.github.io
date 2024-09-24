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

Consider a dataset $\mathcal{D} = \{\mathbf{x}_i\}_{i=1}^n$ where $\mathbf{x} \in \mathbb{R}^d$. Consider an energy based model

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

This is the essence of constrastive divergence. Suppose we do not wish to work with the intractable $Z(\theta)$. Consider then minimizing the Fisher divergence between our proposed distribution $p$ and the data distribution $\hat{p}$:

$$
\begin{aligned}
D_F(p \ ||\  \hat{p}) :&= \mathbb{E}_{\hat{p}(\mathbf{x})}[ \| \nabla_{\mathbf{x}}\log \hat{p}(\mathbf{x}) - \nabla_{\mathbf{x}
}\log p(\mathbf{x})  \|^2] \\ 
&= \mathbb{E}_{\hat{p}(\mathbf{x})}[\|\nabla_{\mathbf{x}}\log \hat{p}(\mathbf{x})\|^2] + \mathbb{E}_{\hat{p}(\mathbf{x})}[\|\nabla_{\mathbf{x}}\log p(\mathbf{x})\|^2 ] - 2 \mathbb{E}_{\hat{p}(\mathbf{x})}[\langle \nabla_{\mathbf{x}}\log \hat{p}(\mathbf{x}), \nabla_{\mathbf{x}
}\log p(\mathbf{x}) \rangle]
\end{aligned}
$$

The first term is constant with respect to $\theta$ and can be dropped during optimization. The second term is dependent on $\theta$ and can be sampled. Looking at the last term:
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
\min_{\theta}\quad\mathbb{E}_{\hat{p}(\mathbf{x})}[\|\nabla_{\mathbf{x}}\log p(\mathbf{x})\|^2  + 2\ \nabla_{\mathbf{x}}^2 \log{p(\mathbf{x})}]
\end{aligned}
$$