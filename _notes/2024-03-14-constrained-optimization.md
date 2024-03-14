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
\end{aligned}
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