---
layout: post
title: "Gaussian Integral"
blurb: "Computing the Gaussian integral by a change of variable"
img: ""
author: "Malav Patel"
categories: journal
tags: []
<!-- image: -->
---

In this problem we are interested in the integral shown below. For convenience we will define the result of the integral as the variable $I$.

$$I = \int_{-\infty}^{\infty} e^{-x^2} \, dx \tag{1}$$

We can instead consider the square of the integral.

$$
\begin{aligned} I^2 &= \bigg(\int_{\mathbb{R}} e^{-x^2} \, dx \bigg)^2 \\\ &= \bigg(\int_{-\infty}^{\infty} e^{-x^2} \, dx\bigg)\bigg(\int_{-\infty}^{\infty} e^{-y^2} \, dy\bigg) \\\ &= \iint\limits_{\mathbb{R}^2} e^{-(x^2 + y^2)} \, dx\,dy  \end{aligned}
$$

Since this integral is over the entirety of $\mathbb{R}^2$, we can alternatively parameterize the integral over the polar coordinates $r$ and $\theta$. The relationships between the variables are shown below. 

$$
\begin{aligned} x &= r \cos{\theta}   &&&  r^2 &= x^2 + y^2\\\ y &= r \sin{\theta}   &&& \theta &= \text{atan2}(y, x) \end{aligned}
$$

<center>
<figure>
  <img src="/assets/img/IMG_7122.PNG">
  <figcaption><i>Polar Coordinates.</i></figcaption>
</figure>
</center>

The differential area element in the rectangular and polar coordinates is given by the following expression. See the following figure for an illustration as to why a factor of $r$ appears.

$$
\begin{aligned}
    dA = dx \, dy = r\,dr\,d\theta
\end{aligned}
$$

<center>
<figure>
  <img src="/assets/img/IMG_7123.JPG">
  <figcaption><i>Polar Coordinates.</i></figcaption>
</figure>
</center>

Using this change of variable our expression for $I^2$ now becomes the following,

$$
\begin{aligned} I^2 &= \int\limits_{0}^{2\pi} \int\limits_{0}^{\infty} e^{-r^2} \,r\,dr\,d\theta \\\ &= 2\pi \Bigg[\frac{1}{2} \int\limits_{0}^{\infty} e^{-u} \, du\Bigg] \\\ &= \pi  \end{aligned}
$$

Where the second line above is due to a u - substitution $ du  = 2r\, dr $. Thus the gaussian integral is given by 

$$ I = \int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi} $$