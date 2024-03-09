---
layout: post
title: "The Least Squares Problem"
blurb: "Deriving and solving the least squares problem in the context of linear regression"
img: ""
author: "Malav Patel"
categories: journal
tags: []
<!-- image: -->
---


To frame the least squares problem, let us consider a dataset $$\mathcal{D} =  \{ \boldsymbol{x}_i, y_i \}_{i=1}^{n}$$. We will assume that the $$y_i$$ are realizations of a random variable $$Y$$ that is a linear function of a non-random variable $$\boldsymbol{x}$$ and zero-mean noise:


\begin{align}
    Y &= \boldsymbol{\theta}^T \boldsymbol{x} + \epsilon &&& \epsilon \sim \mathcal{N}(0, \sigma^2)
\end{align}

The question now is what is the best estimate for $$\boldsymbol{\theta}\,$$? We may consider the following minimization problem which sums up the squared loss of each datapoint from its label:


\begin{equation}
\begin{aligned}
& \underset{\boldsymbol{\theta} \in \mathbb{R}^d}{\text{minimize}}
& & \sum_{i=1}^{n} | y_i - \boldsymbol{\theta}^T\boldsymbol{x}_i |^2 
\end{aligned} 
\end{equation}

If we consider the vector $$\boldsymbol{y} = [y_1,\, y_2,\, ...\, y_n]^T$$ and the matrix $$\boldsymbol{X}$$ with rows given by the $$\boldsymbol{x}_i $$:

\begin{align}
    \boldsymbol{X} = \begin{bmatrix}
    - & \boldsymbol{x}_1^T & -  \\\ 
    - & \boldsymbol{x}_2^T & -  \\\ 
    - &\vdots & - \\\ 
    - & \boldsymbol{x}_n^T & -
\end{bmatrix}
\end{align}

Then the minimization problem takes the following form of a $$\textit{least squares}$$ problem:

\begin{equation}
\begin{aligned}
& \underset{\boldsymbol{\theta} \in \mathbb{R}^d}{\text{minimize}}
& & \\| \boldsymbol{y} - \boldsymbol{X}\boldsymbol{\theta} \\|_2^2 
\end{aligned} \tag{1} \label{1}
\end{equation}

## Solving the Least Squares Problem

Before we solve \eqref{1}, we state the fundamental result of the least-squares problem and its immediate consequences.

<div style="border: 1px solid black; padding: 10px;">
  Let $\boldsymbol{A}$ be an $M$ by $N$ matrix. Then any solution $\hat{\boldsymbol{x}}$ to the least squares problem

  \begin{equation}
  \begin{aligned}
  & \underset{\boldsymbol{x} \in \mathbb{R}^N}{\text{minimize}}
  & & \| \boldsymbol{y} - \boldsymbol{A}\boldsymbol{x} \|_2^2 
  \end{aligned} \tag{2} \label{2}
  \end{equation}

  must obey the $\textbf{normal equations}$:
  \begin{equation}
  \boldsymbol{A}^T\boldsymbol{A}\hat{\boldsymbol{x}} = \boldsymbol{A}^T\boldsymbol{y}
  \end{equation}
</div>

The optimization problem in \eqref{2} is convex, continuous, and differentiable in the variable $$\boldsymbol{x}$$. Because it is convex, a global minimizer to \eqref{2} exists. Because it is continuous and differentiable, we may find the global minimizer by taking the gradient of our objective and setting it equal to zero.


\begin{align}
  \nabla_{\boldsymbol{x}}   \\| \boldsymbol{y} - \boldsymbol{A}\boldsymbol{x} \\|_2^2 \\\ 
  &= \nabla_{\boldsymbol{x}}  
\end{align}


This means

\begin{equation}
     \nabla_{\boldsymbol{x}} \,  \\| \boldsymbol{y} - \boldsymbol{A}\boldsymbol{x} \\|_2^2 = 0 \implies \boldsymbol{A}^T\boldsymbol{A}\boldsymbol{x} - \boldsymbol{A}^T\boldsymbol{y} = 0
\end{equation}


