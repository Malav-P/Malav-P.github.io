---
layout: post
title: "Recommender Systems"
katex: True
blurb: ""
img: ""
author: "Malav Patel"
categories: journal
tags: []
<!-- image: -->
---

What is it? A system that can take in data of users and their interactions, learn a model, and recommend similar interactions. 

What do we need it for? Movie, song, shopping recommendation.


### Matrix Factorization Approach

Assume we have data in the form of a list of tuples $(u, i, r_{ui})$. A tuple tells you that user $u$ gave item $i$ a rating of $r_{ui}$.

We model our guess of the rating as an inner product between a *user factor* $p_u \in \mathbb{R}^d$ and an *item factor* $q_i \in \mathbb{R}^d$ :

$$
\hat{r}_{ui} = q_i^{\top}p_u
$$

And then we do regularized least squares regression, regressing on our list of tuples:

$$
\mathcal{L} = \sum_{r_{ui}}(r_{ui} - \hat{r}_{ui})^2 + \lambda(\|q_i\|^2 + \|p_u\|^2)
$$

The reason it is called matrix factorization is because the model of the user-interaction matrix $\hat{R}$ is modeled as a product of two low-rank matrices:

$$
\hat{R} = Q^{\top}P
$$

where $q_i$ is the $i$-th column of $Q$ and $p_u$ is the $u$-th column of $P$. The term low-rank comes from the fact that both of these matrices are "short and fat" matrices. In other words, the dimension $d \ll$ the number of users and $d \ll$ the number of items. 

### Algorithms for Solving
We typically use stochastic gradient descent to optimize the objective above. Another lesser known method is **alternating least squares**. The basic idea is to freeze $Q$ and solve for $P$ for one iteration and then freeze $P$ and solve for $Q$ in the next iteration. We continue this process until convergence. Notice that by freezing one matrix, the objective becomes a standard least squares problem which we have closed form solutions for. Mathematically we alternate between the first objective optimizing over $Q$


$$
\min_Q \sum_{r_{ui}} \left(r_{ui} - Q_i^\top P_u\right)^2 + \lambda \|Q_i\|^2
$$

and the second objective optimizing over $P$

$$
\min_P \sum_{r_{ui}} \left(r_{ui} - Q_i^\top P_u\right)^2 + \lambda \|P_u\|^2
$$

## SVD

The SVD is a popular matrix decomposition method in linear algebra. In recommender systems it is **not** used the same, but only shares the name because it is similar to the SVD. The SVD approach adds a baseline (i.e. bias terms) to the matrix factorization so that our model becomes:

$$
\hat{r}_{ui} = \underbrace{\mu + b_{u} + b_{i}}_{\text{baseline}} + q_i^{\top}p_u
$$

where $\mu$ is the overall average rating of item $i$ and $b_u$ and $b_i$ are bias parameters for the user and item, respectively. The objective then becomes

$$
\mathcal{L} = \sum_{r_{ui}}(r_{ui} - \hat{r}_{ui})^2 + \lambda(b_i^2 + b_u^2 + \|q_i\|^2 + \|p_u\|^2)
$$