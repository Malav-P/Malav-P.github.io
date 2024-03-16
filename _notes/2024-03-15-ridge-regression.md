---
layout: post
title: "Ridge Regression"
katex: True
blurb: ""
img: ""
author: "Malav Patel"
categories: journal
tags: []
<!-- image: -->
---


<div style="border: 1px solid black; padding: 10px;">

The Ridge Regression problem 
$$
\begin{aligned}
& \underset{\boldsymbol{x} \in \mathbb{R}^d}{\text{minimize}}
& & \| \boldsymbol{y} - \boldsymbol{A}\boldsymbol{x} \|_2^2 + \lambda\lVert \boldsymbol{x} \rVert_2^2
\end{aligned}
$$

is solved uniquely by

$$
\hat{\boldsymbol{x}}_{\text{ridge}} = (\boldsymbol{A}^T\boldsymbol{A} + \lambda\boldsymbol{I})^{-1}\boldsymbol{A}^T\boldsymbol{y}
$$

</div>

Some facts about ridge regression:

- The penalty parameter $\lambda$ controls the trade off between minimizing the error of our linear model and minimizing the norm of our coefficient vector.
- Increasing $\lambda$ tends to shrink the estimate for $\hat{\boldsymbol{x}}$ towards the origin.
- Increasing $\lambda$ introduces *bias* to our estimate but tends to reduce the *variance*.

## Shrinkage
Let $R \leq \text{min}(M, N)$ be the rank of $\boldsymbol{A}$. Consider the *full* SVD of $\boldsymbol{A}$,

$$
\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^T
$$

- $\boldsymbol{U}$ is a $M \times M$ unitary matrix with orthonormal columns that span $\mathbb{R}^M$. Unitarity  implies that $\boldsymbol{U}^T = \boldsymbol{U}^{-1}$.


- $\boldsymbol{\Sigma}$ is a $M \times N$ diagonal matrix of the sorted singular values of the matrix $\boldsymbol{A}.$ ($\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_R \geq 0 $)

$$
  \boldsymbol{\Sigma} =
  \begin{bmatrix}
    \sigma_{1} & & \\
    & \ddots & \\
    & & \sigma_{R}
  \end{bmatrix}
$$

- $\boldsymbol{V}$ is a $N \times N$ matrix with orthonormal columns that span $\mathbb{R}^N$. The orthonormal columns imply that $\boldsymbol{V}^T\boldsymbol{V} = \boldsymbol{I}$. Unitarity  implies that $\boldsymbol{V}^T = \boldsymbol{V}^{-1}$.

Let us look at the ridge regression estimate for $\boldsymbol{y}$:

$$
\begin{aligned}
\hat{\boldsymbol{y}} = \boldsymbol{A}\hat{\boldsymbol{x}}_{\text{ridge}} &= \boldsymbol{A} (\boldsymbol{A}^T\boldsymbol{A} + \lambda\boldsymbol{I})^{-1}\boldsymbol{A}^T\boldsymbol{y} \\\ 

&= \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^T (\boldsymbol{V}\boldsymbol{\Sigma}^T\boldsymbol{\Sigma}\boldsymbol{V}^T + \lambda\boldsymbol{I})^{-1}\boldsymbol{V}\boldsymbol{\Sigma}^T\boldsymbol{U}^T\boldsymbol{y} \\\ 

&= \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^T (\boldsymbol{V}\boldsymbol{\Sigma}^T\boldsymbol{\Sigma}\boldsymbol{V}^T + \boldsymbol{V}\lambda\boldsymbol{I}\boldsymbol{V}^T)^{-1}\boldsymbol{V}\boldsymbol{\Sigma}^T\boldsymbol{U}^T\boldsymbol{y} \\\ 

&= \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^T (\boldsymbol{V}(\boldsymbol{\Sigma}^T\boldsymbol{\Sigma} + \lambda\boldsymbol{I})\boldsymbol{V}^T)^{-1}\boldsymbol{V}\boldsymbol{\Sigma}^T\boldsymbol{U^T}\boldsymbol{y} \\\ 

&= \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^T \boldsymbol{V}(\boldsymbol{\Sigma}^T\boldsymbol{\Sigma} + \lambda\boldsymbol{I})^{-1}\boldsymbol{V}^T\boldsymbol{V}\boldsymbol{\Sigma}^T\boldsymbol{U}^T\boldsymbol{y} \\\ 

&= \boldsymbol{U}\big[\boldsymbol{\Sigma}(\boldsymbol{\Sigma}^T\boldsymbol{\Sigma} + \lambda\boldsymbol{I})^{-1}\boldsymbol{\Sigma}^T\big]\boldsymbol{U}^T\boldsymbol{y} \\\ 

&= \sum_{i=1}^R \boldsymbol{u}_i \frac{\sigma_i^2}{\sigma_i^2 + \lambda} \boldsymbol{u}_i^T\,\boldsymbol{y}

\end{aligned}
$$

We see that the directions with smaller variance (smaller singular values) are shrunk more than those with larger variance.

## Bias-Variance Tradeoff
Here we will show that the ridge estimator is biased but has a lower variance than the ordinary least squares estimator. We now replace the (deterministic) observation vector with a random vector with zero-mean gaussian noise, $\boldsymbol{y} \to \boldsymbol{Y}$.

$$
\begin{aligned}
  &\boldsymbol{Y} = \boldsymbol{Ax} + \boldsymbol{\epsilon}, & & \quad  \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \sigma^2\boldsymbol{I})
\end{aligned}
$$

### Bias
Since the observation vector is now random, the ridge estimator (being a function of this observation vector) is also random. We compute its expectation:

$$
\begin{aligned}
  \mathbb{E}[\hat{\boldsymbol{x}}_{\text{ridge}}] &= \mathbb{E}[(\boldsymbol{A}^T\boldsymbol{A} + \lambda\boldsymbol{I})^{-1}\boldsymbol{A}^T\boldsymbol{Y}] \\\ 
  &= \mathbb{E}[(\boldsymbol{A}^T\boldsymbol{A} + \lambda\boldsymbol{I})^{-1}\boldsymbol{A}^T\boldsymbol{Ax}] + \mathbb{E}[(\boldsymbol{A}^T\boldsymbol{A} + \lambda\boldsymbol{I})^{-1}\boldsymbol{A}^T\boldsymbol{\epsilon}] \\\ 
  &= (\boldsymbol{A}^T\boldsymbol{A} + \lambda\boldsymbol{I})^{-1}\boldsymbol{A}^T\boldsymbol{Ax} + (\boldsymbol{A}^T\boldsymbol{A} + \lambda\boldsymbol{I})^{-1}\boldsymbol{A}^T\,\mathbb{E}[\boldsymbol{\epsilon}] \\\ 
  &= (\boldsymbol{A}^T\boldsymbol{A} + \lambda\boldsymbol{I})^{-1}\boldsymbol{A}^T\boldsymbol{Ax} \\\ 
  &\neq \boldsymbol{x}
\end{aligned}
$$

We see that the ridge regression estimator is biased when $\lambda \neq 0$.

### Variance