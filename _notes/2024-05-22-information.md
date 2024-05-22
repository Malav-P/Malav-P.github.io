---
layout: post
title: "Information"
katex: True
blurb: ""
img: ""
author: "Malav Patel"
categories: journal
tags: []
<!-- image: -->
---

Suppose we have a continuous random variable $X$ taking values in a set $\mathcal{X}$ and is distributed according to $p : \mathcal{X} \to [0, 1]$.

The information of a particular realization of $X$, denoted by $x \in \mathcal{X}$, is defined as
$$
\begin{aligned}
I(x) &= \text{log}_2\Big(\frac{1}{p(x)}\Big) \\
     &= -\text{log}_2\Big(p(x)\Big)
\end{aligned}
$$

We see that for events $x$ that have low probability, the information is high, reflecting the fact that we are surprised to see a low probability event. In contrast, events with high probability have low information, reflecting the fact that we are unsurprised by observing an event that happens often. 

In essence, $\textbf{low probability events} \to \textbf{high information}.$

The expected value of the information is called the entropy, $H(X)$:

$$
\begin{aligned}
H(X) &\triangleq \mathbb{E}(I) \\ 
&= \mathbb{E}[-\text{log}_2(p(X))] \\
&= \sum_{x \in \mathcal{X}} -p(x)\ \text{log}_2(p(x))
\end{aligned}
$$

Where the expectation above is taken with respect to $X$.