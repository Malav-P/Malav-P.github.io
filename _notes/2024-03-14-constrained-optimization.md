---
layout: post
title: "Constrained Optimization"
katex: True
blurb: ""
img: ""
author: "Malav Patel"
categories: journal
tags: []
<!-- image: -->
---

## Problem

$$
\begin{aligned}
 \underset{\boldsymbol{x} \in \mathbb{R}^{n}}{\text{minimize}}& \quad f(\boldsymbol{x}) & & \quad f:\mathbb{R}^n \to \mathbb{R} \\\ 
 \text{s.t.}& \quad \boldsymbol{c}(\boldsymbol{x}) \leq 0 & & \quad \boldsymbol{c} : \mathbb{R}^n \to \mathbb{R}^m
\end{aligned} \tag{1}
$$

Above, we assume both $f$ and $\boldsymbol{c}$ are continuous and differentiable. Define the *constraint Jacobian*:

$$
\begin{aligned}
    \boldsymbol{A} = \begin{bmatrix}
    \nabla c_1 & | &  \nabla c_2 & | & \cdots & | & \nabla c_m
\end{bmatrix}
\end{aligned}
$$

Define the *Active Constraint Set* which returns the indices of the constraints that are active at a feasible point $\boldsymbol{x}$:

$$
\mathcal{A}(\boldsymbol{x}) \triangleq \{i : c_i(\boldsymbol{x}) = 0\}
$$

Define the set of *descent* directions at a point $\boldsymbol{x}$ as the set of unit-step directions which result in a descrease in the objective function's value:

$$
\mathcal{D}(\boldsymbol{x}) \triangleq \{\boldsymbol{d} \in \mathbb{R}^n : \boldsymbol{d}^T\,\nabla f(\boldsymbol{x}) < 0, \quad \lVert \boldsymbol{d}\rVert_2 = 1\}
$$

Define the set of all *feasible directions* as the set of step directions that result in a new point that satisfies our constraints. At a point $\boldsymbol{x}$ we only care about not violating the constraints that are active: we want to move in a direction orthogonal to the active constraint gradients or in a direction opposite to them:

$$
\mathcal{F}(\boldsymbol{x}) = \{\boldsymbol{d}\in \mathbb{R}^n : \boldsymbol{d}^T \, \nabla c_i(\boldsymbol{x}) \leq 0 \, \, \, \,\forall\, i \in \mathcal{A}(\boldsymbol{x}), \quad \lVert \boldsymbol{d}\rVert_2 = 1\}
$$

Note that if $\boldsymbol{x}^{\*}$ is a local optimal solution of (1), then we must have $\mathcal{D}(\boldsymbol{x}^{\*})\, \cap\, \mathcal{F}(\boldsymbol{x}^{\*}) = \empty$. In other words, there are no directions that we may step from $\boldsymbol{x}^{\*}$ that simultaneously reduce the value of $f$ and keep the new point feasible. This can be easily proved by contradiction:
- Proof <br>
  Suppose that $\mathcal{D}(\boldsymbol{x}^{\*})\, \cap\, \mathcal{F}(\boldsymbol{x}^{\*}) \neq \empty$. Then we can choose a direction $\boldsymbol{d} \in \mathcal{D}(\boldsymbol{x}^{\*})\, \cap\, \mathcal{F}(\boldsymbol{x}^{\*})$. Let $\bar{\boldsymbol{x}} = \boldsymbol{x}^{\*} + \alpha\boldsymbol{d}$ for a small value of $\alpha > 0$. Then $\bar{\boldsymbol{x}}$ is feasible and $f(\bar{\boldsymbol{x}}) \leq f(\boldsymbol{x}^{\*})$ since $\boldsymbol{d}$ is both a feasible and descent direction. This violates our initial assumption that $\boldsymbol{x}^{\*}$ was the local minimizer.

## Towards the KKT Conditions
At $\boldsymbol{x}^{\*}$ we must have

$$
\begin{aligned}
&\mathcal{D}(\boldsymbol{x}^{*})\, \cap\, \mathcal{F}(\boldsymbol{x}^{*}) = \empty 
\end{aligned} \tag{2}
$$

From this, we can intuit a form for the gradient of our objective at the optimizer:

$$
\begin{aligned}
   \nabla f(\boldsymbol{x}^{*}) &= -\sum_{i \in \mathcal{A}(\boldsymbol{x}^{*})} \lambda_i \, \nabla c_i(\boldsymbol{x}^{*}), & & \lambda_i \geq0
\end{aligned}
$$

This says that the gradient is a linear combination of the active constraint gradients at $\boldsymbol{x}^{\*}$. To see that such a form is consistent with (2), let us show that an element of $\mathcal{D}(\boldsymbol{x}^{\*})$ cannot be an element of $\mathcal{F}(\boldsymbol{x}^{\*})$ and vice versa:

#### I. $\boldsymbol{d} \in \mathcal{F}(\boldsymbol{x}^{*}) \implies d \notin \mathcal{D}(\boldsymbol{x}^{*})$ 

Let us project $\boldsymbol{d}$ onto the gradient,

$$
\begin{aligned}
  \boldsymbol{d}^T\,\nabla f(\boldsymbol{x}^{*}) 
  &= \sum_{i \in \mathcal{A}(\boldsymbol{x}^{*})}-\lambda_i \, \boldsymbol{d}^T \nabla c_i(\boldsymbol{x}^{*}) \\\ 
  &\geq 0
\end{aligned}
$$

Where the last equality follows since $\lambda_i \geq 0$ and $\boldsymbol{d}^T \nabla c_i(\boldsymbol{x}^{\*}) \leq0$.

#### II. $\boldsymbol{d} \in \mathcal{D}(\boldsymbol{x}^{\*}) \implies d \notin \mathcal{F}(\boldsymbol{x}^{\*})$

Let us consider the projection of $\boldsymbol{d}$ onto the gradient again:

$$
\begin{aligned}
  \boldsymbol{d}^T\,\nabla f(\boldsymbol{x}^{*}) 
  &= \sum_{i \in \mathcal{A}(\boldsymbol{x}^{*})} -\lambda_i \, \boldsymbol{d}^T \nabla c_i(\boldsymbol{x}^{*})
  \leq 0
\end{aligned} \tag{3}
$$

Since $\lambda_i \geq 0$, there exists at least one $j$ such that $\boldsymbol{d}^T \nabla c_j(\boldsymbol{x}^{\*}) > 0$ so that (3) is satisfied. As a result, $ d \notin \mathcal{F}(\boldsymbol{x}^{\*})$.