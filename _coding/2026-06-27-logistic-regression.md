---
layout: post
title: "Logistic Regression"
katex: False
blurb: ""
img: ""
author: "Malav Patel"
categories: journal
tags: []
<!-- image: -->
---

Prompt: Implement Logistic Regression. 
Assume X_train contains a set of N d-dimensional data points.

## Gradient Calculation

We start with the Bernoulli distribution with parameter $p$. The probability mass function is given by:

$$
    P(Y=y) = p^y(1-p)^{1-y}
$$

In logistic regression, we let the probability parameter be a function of our data, $x$. Specifically, we allow p to be the composition of two transformations: first any differentiable function $f_{\theta}$ followed by the sigmoid function. The range of the sigmoid function is (0, 1), ensuring that our probability parameter $p$ is indeed a probability.

$$
\begin{aligned}
p :&= \sigma(f_{\theta}(x)) \\ 
  &= \frac{1}{1 + e^{-f_{\theta}(x)}}
\end{aligned}
$$

Plugging this into our density, we can derive our negative-log-likelihood, which will be the loss function for logistic regression:

$$
- \ell(\theta) = y \log \sigma + (1-y)\log (1 - \sigma)
$$

We can now take the gradient:

$$
\begin{aligned}
-\frac{d\ell}{d\theta} &= \frac{y}{\sigma}\frac{d\sigma}{df}\frac{df}{d\theta} - \frac{1-y}{1-\sigma}\frac{d\sigma}{df}\frac{df}{d\theta} \\ 

&= \bigg[\frac{y}{\cancel{\sigma}}\cancel{\sigma}(1-\sigma) - \frac{1-y}{\cancel{1-\sigma}}\sigma\cancel{(1-\sigma)}\bigg] \frac{df}{d\theta}  \\ 

&= (y - \sigma)\frac{df}{d\theta}


\end{aligned}
$$


Suppose $f_\theta(x) := w^\top x + b$. We can rewrite this as a single dot product by letting

$$x \gets \begin{bmatrix} x \\ 1 \end{bmatrix}, \qquad \theta := \begin{bmatrix} w \\ b \end{bmatrix}$$

Then we have the equivalent expression 

$$
f_\theta(x) = \theta^\top x
$$

and then our loss gradient is simply

$$
\frac{d\ell}{d\theta} = (\sigma - y) x
$$


## Coding It Up

```python
def sigma(x):
    """
    Applies the sigmoid function to x

    Args:
        x : np array 

    Returns:
        np array where elementwise we have 1 / 1 + exp(-x)
    """

    return 1 / (1+np.exp(-x))


class LogisticRegression:
    def __init__(self):
        self.theta = None # (d + 1)
        self.d = None

    def fit(self, X_train, y_train, alpha=1e-3, num_iters=100):
        """
        Args:
            X_train: np array of shape (N, d)
            y_train: np array of shape (N,)

        Returns:
            theta: np array of shape (d+1,), the fitted parameters
        """

        N, d = X_train.shape

        self.d = d
        self.theta = np.random.normal(size=d+1)

        X = np.column_stack((X_train, np.ones(N))) # (N, d+1)


        for _ in range(num_iters):
            diff = sigma(X @ self.theta) - y_train # (N,)
            grad = 1 / N * (diff @ X) # (d+1,)

            self.theta -= alpha * grad

        
        return self.theta    
```