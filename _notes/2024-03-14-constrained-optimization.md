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
    - & \nabla c_1^T & -  \\[1mm]
    - & \nabla c_2^T & -  \\[1mm]
    - &\vdots & - \\[1mm]
    - & \nabla c_m^T & -
\end{bmatrix}
\end{aligned}
$$

Define the *Active Constraint Set* which returns the indices of the constraints that are active at a feasible point $\boldsymbol{x}$:

$$
\mathcal{A}(\boldsymbol{x}) \triangleq \{i : c_i(\boldsymbol{x}) = 0\}
$$

Define the set of *descent* directions at a point $\boldsymbol{x}$ as the set of unit-step directions which result in a descrease in the objective function's value:

$$
\mathcal{D}(\boldsymbol{x}) \triangleq \{\boldsymbol{d} \in \mathbb{R}^n : \boldsymbol{d}^T\,\nabla f(\boldsymbol{x}) \leq 0, \quad \lVert \boldsymbol{d}\rVert_2 = 1\}
$$

Define the set of all *feasible directions* as the set of step directions that result in a new point that satisfies our constraints. At a point $\boldsymbol{x}$ we only care about not violating the constraints that are active: we want to move in a direction orthogonal to the active constraint gradients or in a direction opposite to them:

$$
\mathcal{F}(\boldsymbol{x}) = \{\boldsymbol{d}\in \mathbb{R}^n : \boldsymbol{d}^T \, \nabla c_i(\boldsymbol{x}) \leq 0 \, \, \, \,\forall\, i \in \mathcal{A}(\boldsymbol{x}), \quad \lVert \boldsymbol{d}\rVert_2 = 1\}
$$

Note that if $\boldsymbol{x}^*$ is a local optimal solution of (1), then we must have $\mathcal{D}(\boldsymbol{x}^*)\, \cap\, \mathcal{F}(\boldsymbol{x}^*) = \empty$. In other words, there are no directions that we may step from $\boldsymbol{x}^*$ that simultaneously reduce the value of $f$ and keep the new point feasible. This can be easily proved by contradiction:
- Proof <br>
  Suppose that $\mathcal{D}(\boldsymbol{x}^*)\, \cap\, \mathcal{F}(\boldsymbol{x}^*) \neq \empty$. Then we can choose a direction $\boldsymbol{d} \in \mathcal{D}(\boldsymbol{x}^*)\, \cap\, \mathcal{F}(\boldsymbol{x}^*)$. Let $\bar{\boldsymbol{x}} = \boldsymbol{x}^* + \alpha\boldsymbol{d}$ for a small value of $\alpha > 0$. Then $\bar{\boldsymbol{x}}$ is feasible and $f(\bar{\boldsymbol{x}}) \leq f(\boldsymbol{x}^*)$ since $\boldsymbol{d}$ is both a feasible and descent direction. This violates our initial assumption that $\boldsymbol{x}^*$ was the local minimizer.

## Towards the KKT Conditions
At $\boldsymbol{x}^*$ we must have $\mathcal{D}(\boldsymbol{x}^*)\, \cap\, \mathcal{F}(\boldsymbol{x}^*) = \empty$. 