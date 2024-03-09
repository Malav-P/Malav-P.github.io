---
layout: post
title: "The Least Squares Problem"
blurb: "Deriving and solving the least square problem in the context of linear regression"
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
    - & \boldsymbol{x}_n^T & - \\\
\end{bmatrix}
\end{align}

Then the minimization problem takes the following form of a $$\textit{least squares}$$ problem:

\begin{equation}
\begin{aligned}
& \underset{\boldsymbol{\theta} \in \mathbb{R}^d}{\text{minimize}}
& & \| \boldsymbol{y} - \boldsymbol{X}\boldsymbol{\theta} \|_2^2 
\end{aligned} \tag{1}
\end{equation}