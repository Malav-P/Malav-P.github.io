---
layout: post
title: "The Least Squares Problem"
katex: True
blurb: "Deriving and solving the least squares problem in the context of linear regression"
img: ""
author: "Malav Patel"
categories: journal
tags: []
<!-- image: -->
---


To frame the least squares problem, let us consider a dataset $\mathcal{D} =  \\{ \boldsymbol{x}\_i, y_i \\}\_{i=1}^{n}$. We will assume that the $y_i$ are realizations of a random variable $Y$ that is a linear function of a non-random variable $\boldsymbol{x}$ and zero-mean noise:

$$
\begin{aligned}
    Y &= \boldsymbol{\theta}^T \boldsymbol{x} + \epsilon &&& \epsilon \sim \mathcal{N}(0, \sigma^2)
\end{aligned}
$$

The question now is what is the best estimate for $\boldsymbol{\theta}\,$? We may consider the following minimization problem which sums up the squared loss of each datapoint from its label:


$$
\begin{aligned}
& \underset{\boldsymbol{\theta} \in \mathbb{R}^d}{\text{minimize}}
& & \sum_{i=1}^{n} | y_i - \boldsymbol{\theta}^T\boldsymbol{x}_i |^2 
\end{aligned} 
$$

If we consider the vector $\boldsymbol{y} = [y_1,\, y_2,\, ...\, y_n]^T$ and the matrix $\boldsymbol{X}$ with rows given by the $\boldsymbol{x}_i $:

$$
\begin{aligned}
    \boldsymbol{X} = \begin{bmatrix}
    - & \boldsymbol{x}_1^T & -  \\\ 
    - & \boldsymbol{x}_2^T & -  \\\ 
    - &\vdots & - \\\ 
    - & \boldsymbol{x}_n^T & -
\end{bmatrix}
\end{aligned}
$$

Then the minimization problem takes the following form of a $\textit{least squares}$ problem:

$$
\begin{aligned}
& \underset{\boldsymbol{\theta} \in \mathbb{R}^d}{\text{minimize}}
& & \| \boldsymbol{y} - \boldsymbol{X}\boldsymbol{\theta} \|_2^2 
\end{aligned} \tag{1}
$$

## Solving the Least Squares Problem

Before we solve (1), we state the fundamental result of the least-squares problem and its immediate consequences.

<div style="border: 1px solid black; padding: 10px;">
  Let $\boldsymbol{A}$ be an $M$ by $N$ matrix. Then any solution $\hat{\boldsymbol{x}}$ to the least squares problem

  $$
  \begin{aligned}
  & \underset{\boldsymbol{x} \in \mathbb{R}^N}{\text{minimize}}
  & & \| \boldsymbol{y} - \boldsymbol{A}\boldsymbol{x} \|_2^2 
  \end{aligned} \tag{2}
  $$

  must obey the $\textbf{normal equations}$:
  $$
  \boldsymbol{A}^T\boldsymbol{A}\hat{\boldsymbol{x}} = \boldsymbol{A}^T\boldsymbol{y} \tag{3}
  $$
</div>

The optimization problem in (2) is convex, continuous, and differentiable in the variable $\boldsymbol{x}$. Because it is convex, a global minimizer to (2) exists. Because it is continuous and differentiable, we may find the global minimizer by taking the gradient of our objective and setting it equal to zero.

$$
\begin{aligned}
 \nabla_{\boldsymbol{x}}  \| \boldsymbol{y} - \boldsymbol{A}\boldsymbol{x} \|_2^2 &= \nabla_{\boldsymbol{x}} \big(\boldsymbol{y} - \boldsymbol{A}\boldsymbol{x} \big)^T \big(\boldsymbol{y} - \boldsymbol{A}\boldsymbol{x} \big) \\\ 
 &= \nabla_{\boldsymbol{x}} \big(\boldsymbol{y}^T\boldsymbol{y} - 2\boldsymbol{x}^T\boldsymbol{A}^T\boldsymbol{y} + \boldsymbol{x}^T\boldsymbol{A}^T\boldsymbol{A}\boldsymbol{x} \big) \\\ 
 &= 2\boldsymbol{A}^T\boldsymbol{A}\boldsymbol{x} - 2\boldsymbol{A}^T\boldsymbol{y} 
\end{aligned}
$$

This means

$$
  \nabla_{\boldsymbol{x}} \,  \| \boldsymbol{y} - \boldsymbol{A}\boldsymbol{x} \|_2^2 = 0 \implies \boldsymbol{A}^T\boldsymbol{A}\boldsymbol{x} - \boldsymbol{A}^T\boldsymbol{y} = 0
$$

A couple of immediate consequences we can see are:

<!-- $$
\begin{itemize} -->
- A solution to (3) always exists. Since the vector $\boldsymbol{A}^T\boldsymbol{y}$ $\in$ Row($\boldsymbol{A}$), and it is a fact that Row($\boldsymbol{A}$) $\triangleq$ Col($\boldsymbol{A}^T$) = Col($\boldsymbol{A}^T\boldsymbol{A}$), we can conclude that $\boldsymbol{A}^T\boldsymbol{y}$ $\in$ Col($\boldsymbol{A}^T\boldsymbol{A}$). In other words, we can express the vector $\boldsymbol{A}^T\boldsymbol{y}$ as a linear combination of the columns of $\boldsymbol{A}^T\boldsymbol{A}$, i.e. there exists at least one $\hat{\boldsymbol{x}}$ such that $\boldsymbol{A}^T\boldsymbol{A}\hat{\boldsymbol{x}} = \boldsymbol{A}^T\boldsymbol{y}$.
- In the case that rank($\boldsymbol{A}$) = $N$, the square matrix $\boldsymbol{A}^T\boldsymbol{A}$ has full rank and hence is invertible. There exists one unique solution to (3) given by 

$$
\hat{\boldsymbol{x}} = \big(\boldsymbol{A}^T\boldsymbol{A}\big)^{-1}\boldsymbol{A}^T\boldsymbol{y}
$$

- In the case that rank($\boldsymbol{A}$) $<$ $N$, there are infinitely many solutions to (3) since the null space of $\boldsymbol{A}^T\boldsymbol{A}$ is nontrivial. To see this, consider a solution $\hat{\boldsymbol{x}}$ to (3) and define a vector $\hat{\boldsymbol{z}} = \hat{\boldsymbol{x}} + \boldsymbol{x}\_{\text{null}}$, where $\boldsymbol{x}\_{\text{null}} \in \text{Null}(\boldsymbol{A}^T\boldsymbol{A})$. Then $\hat{\boldsymbol{z}}$ is also a solution of (3):

$$
\begin{aligned}
    \boldsymbol{A}^T\boldsymbol{A}\hat{\boldsymbol{z}} &= \boldsymbol{A}^T\boldsymbol{A}(\hat{\boldsymbol{x}} + \boldsymbol{x}_{\text{null}}) \\\ 
    &= \boldsymbol{A}^T\boldsymbol{A}\hat{\boldsymbol{x}} + \boldsymbol{A}^T\boldsymbol{A}\boldsymbol{x}_{\text{null}} \\\ 
    &= \boldsymbol{A}^T\boldsymbol{y} + \boldsymbol{0} \\\ 
    &= \boldsymbol{A}^T\boldsymbol{y}
\end{aligned}
$$

- In the case that rank($\boldsymbol{A}$) = $M$, there exists at least one $\hat{\boldsymbol{x}}$ that satisfies $\boldsymbol{A}\hat{\boldsymbol{x}} = \boldsymbol{y}$. Note that this solution satifies (3) and achieves the smallest possible objective value: $\\|\boldsymbol{y} - \boldsymbol{A}\hat{\boldsymbol{x}}\\|_2^2 = 0$.
<!-- \end{itemize}
$$ -->

### A Universal Solution via the Singular Value Decomposition

Consider the singular value decomposition of our matrix $\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^T$. Let $R =$ rank($\boldsymbol{A}$). As a quick recap, the properties of the three matrices are outlined below.

- $\boldsymbol{U}$ is a $M \times R$ matrix with orthonormal columns that span the column space of $\boldsymbol{A}$. The orthonormal columns imply that $\boldsymbol{U}^T\boldsymbol{U} = \boldsymbol{I}$ (note that generally $\boldsymbol{U}\boldsymbol{U}^T \neq \boldsymbol{I}$)

$$
\begin{aligned}
    \boldsymbol{U} = \begin{bmatrix}
      | & | &  & | \\\ 
      | & | &  & | \\\ 
      | & | &  & | \\\ 
      \boldsymbol{u}_1 & \boldsymbol{u}_2 & \cdots & \boldsymbol{u}_R \\\
      | & | &  & | \\\
      | & | &  & | \\\ 
      | & | &  & |
\end{bmatrix}
\end{aligned}
$$

- $\boldsymbol{\Sigma}$ is a $R \times R$ diagonal matrix of the singular values of the matrix $\boldsymbol{A}$.

$$
  \boldsymbol{\Sigma} =
  \begin{bmatrix}
    \sigma_{1} & & \\
    & \ddots & \\
    & & \sigma_{R}
  \end{bmatrix}
$$

- $\boldsymbol{V}$ is a $N \times R$ matrix with orthonormal columns that span the row space of $\boldsymbol{A}$. The orthonormal columns imply that $\boldsymbol{V}^T\boldsymbol{V} = \boldsymbol{I}$ (note that generally $\boldsymbol{V}\boldsymbol{V}^T \neq \boldsymbol{I}$)

$$
\begin{aligned}
    \boldsymbol{V}^T = \begin{bmatrix}
    ---- & \boldsymbol{v}_1^T & ---- \\\ 
    ---- & \boldsymbol{v}_2^T & ----  \\\ 
     &\vdots &  \\\ 
    ---- & \boldsymbol{v}_R^T & ----
\end{bmatrix}
\end{aligned}
$$

