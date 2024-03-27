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
To compute the variance we take note of the following assumptions/facts:

- Assumption : The matrix $\boldsymbol{A}^T\boldsymbol{A}$ is invertible and thus, the ordinary least squares (OLS) estimator $\hat{\boldsymbol{\theta}}_{\text{OLS}} = (\boldsymbol{A}^T\boldsymbol{A})^{-1}\boldsymbol{A}^T\boldsymbol{y}$ exists.
- Fact : if $X$ is a random vector with covariance matrix $\text{cov}(X) = \boldsymbol{\Sigma}$, then the random vector formed via linear transformation $Y = \boldsymbol{C}X$ has covariance matrix $\text{cov}(Y) = \boldsymbol{C}\boldsymbol{\Sigma}\boldsymbol{C}^T$
- Fact : The inverse of the transpose of a matrix $\boldsymbol{C}$ equals the transpose of its inverse : $(\boldsymbol{C}^T)^{-1} = (\boldsymbol{C}^{-1})^T$


$$
\begin{aligned}
  \text{Var}[\hat{\boldsymbol{x}}_{\text{ridge}}] &= \text{Var}[(\boldsymbol{A}^T\boldsymbol{A} + \lambda\boldsymbol{I})^{-1}\boldsymbol{A}^T\boldsymbol{Y}] \\\ 
  &= \text{Var}[(\boldsymbol{A}^T\boldsymbol{A} + \lambda\boldsymbol{I})^{-1}\boldsymbol{A}^T\boldsymbol{A}\underbrace{(\boldsymbol{A}^T\boldsymbol{A})^{-1}\boldsymbol{A}^T\boldsymbol{Y}}_{\hat{\boldsymbol{x}}_{\text{OLS}}}] \\\ 
  &= (\boldsymbol{A}^T\boldsymbol{A} + \lambda\boldsymbol{I})^{-1}\boldsymbol{A}^T\boldsymbol{A}\text{Var}[\hat{\boldsymbol{x}}_{\text{OLS}}]\boldsymbol{A}^T\boldsymbol{A}(\boldsymbol{A}^T\boldsymbol{A} + \lambda\boldsymbol{I})^{-1} \\\ 
  &= \sigma^2(\boldsymbol{A}^T\boldsymbol{A} + \lambda\boldsymbol{I})^{-1}\boldsymbol{A}^T\boldsymbol{A}(\boldsymbol{A}^T\boldsymbol{A} + \lambda\boldsymbol{I})^{-1}
\end{aligned}
$$

In the last line, we use the result for the variance of the OLS estimator. To show that the OLS estimator has higher variance than the ridge estimator, we must show that along any direction vector, the variance of the OLS is larger than the variance of the ridge estimator. The variance along any particular direction $\boldsymbol{u}$ is given by $\boldsymbol{u}^T$ the difference matrix $\text{Var}[\hat{\boldsymbol{x}}_{\text{OLS}}] - \text{Var}[\hat{\boldsymbol{x}}_{\text{ridge}}]$ is *positive definite*.

### Variance Along a Particular Direction
To show that the OLS estimator has higher variance than the ridge estimator, we must take a brief aside to clear define what it means for a random vector $\boldsymbol{X}$ to have higher variance than a random vector $\boldsymbol{Y}$. Consider the 2D-case of a random vector $\boldsymbol{X}$ and its covariance matrix $\Sigma$. 

$$
\begin{aligned}
\Sigma = \begin{bmatrix}
\sigma_1^2 & \sigma_{12} \\\ 
\sigma_{21} & \sigma_2^2

\end{bmatrix}
\end{aligned}

$$

In this case, $\sigma_1^2$ captures the variance along the x-axis (i.e. along the vector $\boldsymbol{u} = [1\quad 0]^T$). Similalry $\sigma_2^2$ captures the variance along the y axis (i.e. along the vector $\boldsymbol{u} = [0\quad 1]^T$).