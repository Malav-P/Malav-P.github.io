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

Since this integral is over the entirety of $\mathbb{R}^2$, we can alternatively parameterize the integral over the polar coordinates $r$ and $\phi$. The relationships between the variables are shown below. 

\begin{align} x &= r \cos{\phi}   &&&  r^2 &= x^2 + y^2\\\ y &= r \sin{\phi}   &&& \phi &= \text{atan2}(y, x) \end{align}

<center>
<figure>
  <img src="/assets/img/IMG_7122.PNG">
  <figcaption><i>Polar Coordinates.</i></figcaption>
</figure>
</center>


### Header 3

