---
layout: post
title: "Rectified Flow"
katex: True
blurb: ""
img: ""
author: "Malav Patel"
categories: journal
tags: []
<!-- image: -->
---

What is it? A method for learning a map from a tractable distribution to an untractable (the observed data) distribution.

What do we need it for? In order to have a generative model from which we can sample more data points from.


### Problem setup

We are given a random variable $x_0$ from a tractable distribution (e.g. gaussian) and a random variable $x_1$ from an intractable distribution (i.e. the data distribution from which we have a set of samples). Define a linear interpolation between $x_0$ and $x_1$:

$$
x_t = tx_1 + (1-t)x_0
$$

The above equation can be seen as the solution to the ODE:

$$
\frac{d x_t}{dt} = x_1 - x_0 \triangleq v(x_1, x_0)
$$

with initial condition $x_0$ at $t=0$. 

#### What we want to do
1. sample $x_0 \sim \mathcal{N}(0, I)$
2. numerically integrate forward in time by a small amount, $x_{\Delta t} = x_0 + v(x_1, x_0) \Delta t$
3. numerically integrate again, $x_{2\Delta t} = x_{\Delta t} + v(x_1, x_0) \Delta t$
4. $\vdots$
5. Keep integrating until we get to $x_1$

But, we get stuck at step 2, because we dont have $x_1$ to plug into $v(x_1, x_0)$. This is the problem of causality.

### Solution
Model the velocity field by a neural network and make it causal by replacing $x_1$ with $x_t$.

$$
\frac{dx_t}{dt} \approx v_{\theta}(x_t, x_0)
$$

We can learn the velocity field approximator $v_{\theta}$ by regressing against its target value:

$$
\mathcal{L} = \mathbb{E}[\|(x_1 - x_0) - v_{\theta}(x_t, x_0)\|_2^2]
$$


### Learning the Model
A simple algorithm would be


-  $x_1\gets$ get datapoint from dataset
-  $x_0 \sim \mathcal{N}(0, I)$
-  $ t \sim U(0, 1)$
-  $x_t \gets tx_1 + (1-t)x_0$
- $\mathcal{L} \gets\|(x_1 - x_0) - v_{\theta}(x_t, x_0)\|_2^2$
- Take gradient, $\nabla_{\theta}\mathcal{L}$ 
- Take a gradient step
   - $\theta \gets \theta - \alpha\nabla_{\theta}\mathcal{L}$
- Repeat until converged


To get less noisy estimates of the gradient, we would normally do minibatches of $x_1$.

### After We've Learned
After we have learned our velocity field model, we can generate samples by doing the following:

1. sample $x_0 \sim \mathcal{N}(0, I)$
2. numerically integrate forward in time by a small amount, $x_{\Delta t} = x_0 + v_{\theta}(x_{\Delta t}, x_0) \Delta t$
3. numerically integrate again, $x_{2\Delta t} = x_{\Delta t} + v_{\theta}(x_{\Delta t}, x_0) \Delta t$
4. $\vdots$
5. Keep integrating until we get to $x_1$






