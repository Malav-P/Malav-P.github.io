---
layout: post
title: "The Gaussian Integral"
blurb: "Computing the Gaussian integral by a change of variable"
img: ""
author: "Malav Patel"
categories: journal
tags: []
<!-- image: -->
---

In this problem we are interested in the integral shown below. For convenience we will define the result of the integral as the variable $I$.

\begin{equation} I = \int_{-\infty}^{\infty} e^{-x^2} \, dx \tag{1} \end{equation}

We can instead consider the square of the integral.

\begin{align} I^2 &= \bigg(\int_{\mathbb{R}} e^{-x^2} \, dx \bigg)^2 \\\ &= \bigg(\int_{-\infty}^{\infty} e^{-x^2} \, dx\bigg)\bigg(\int_{-\infty}^{\infty} e^{-y^2} \, dy\bigg) \\\ &= \iint\limits_{\mathbb{R}^2} e^{-(x^2 + y^2)} \, dx\,dy  \end{align}

Since this integral is over the entirety of $\mathbb{R}^2$, we can alternatively parameterize the integral over the polar coordinates $r$ and $\theta$. The relationships between the variables are shown below. 

\begin{align} x &= r \cos{\theta}   &&&  r^2 &= x^2 + y^2\\\ y &= r \sin{\theta}   &&& \theta &= \text{atan2}(y, x) \end{align}

<center>
<figure>
  <img src="/assets/img/IMG_7122.PNG">
  <figcaption><i>Polar Coordinates.</i></figcaption>
</figure>
</center>

The differential area element in the rectangular and polar coordinates is given by the following expression. See the following figure for an illustration as to why a factor of $r$ appears.

\begin{align}
    dA = dx \, dy = r\,dr\,d\theta
\end{align}

<center>
<figure>
  <img src="/assets/img/IMG_7123.JPG">
  <figcaption><i>Polar Coordinates.</i></figcaption>
</figure>
</center>

Using this change of variable our expression for $I^2$ now becomes the following,

\begin{align} I^2 &= \int\limits_{0}^{2\pi} \int\limits_{0}^{\infty} e^{-r^2} \,r\,dr\,d\theta \\\ &= 2\pi \Bigg[\frac{1}{2} \int\limits_{0}^{\infty} e^{-u} \, du\Bigg] \quad \text{Use u-substitution} u = r^2 \\\ &= \pi  \end{align}

Thus the gaussian integral is given by 


\begin{equation} I = \int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi} \end{equation}





### Header 3

