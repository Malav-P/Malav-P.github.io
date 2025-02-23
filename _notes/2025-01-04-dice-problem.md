---
layout: post
title: "Dice Problem"
blurb: "Average Number of Rolls Needed to get a specific outcome when rolling a die"
img: ""
author: "Malav Patel"
categories: journal
tags: []
<!-- image: -->
---

Consider the following question: Suppose you have a single fair die. What is the average number of rolls needed to roll two 6's in a row? What about a 6 followed by a 5?

## Answer via Expected Value

### Two 6's in a Row

Let $X$ be the random variable describing the number of times we roll before we get two 6's in a row. We are interested in the average number of rolls, $\mathbb{E}[X]$. During our first roll, we have two possible events that can occur:

- Event $A:$ first roll is 6
- Event $B:$ first roll is not a 6

We can use the law of total expectation to write:

$$
\begin{aligned}
\mathbb{E}[X] = P(\text{A})\cdot\mathbb{E}[X | \text{A}] + P(\text{B})\cdot\mathbb{E}[X | B]
\end{aligned}
$$

Let us now continue to work on this equation. Since the die is fair, we have $P(\text{A}) = \frac{1}{6}$ and $P(B)=\frac{5}{6}$. Now consider $\mathbb{E}[X \| A]$, which describes the average number of rolls needed to get two 6's given that the previous roll was a 6. We have 2 possibilities in this scenario:

- Event $C$: our next roll is a 6 and the game ends. This happens with probability $\frac{1}{6}$. So we have $P(C) = \frac{1}{6}$. Our cost is 1 roll for event $C$ and 1 roll for event $A$, so we have $\mathbb{E}[X \| C, A] = 2$.
- Event $D$: our next roll is not a 6 we are back to square one. This happens with probability $\frac{5}{6}$. Since we used one roll and ended up back at square 1, we have wasted 1 roll in event $D$ and used 1 roll in event $A$, so $\mathbb{E}[X \| D, A] = 2 + \mathbb{E}[X]$. 

We can use the law of total expectation again to write

$$
\begin{aligned}
\mathbb{E}[X | A] &= P(\text{C})\cdot\mathbb{E}[X |C, A] + P(D)\cdot\mathbb{E}[X | D,A] \\ 
&= \frac{1}{6}\cdot (2) + \frac{5}{6}\cdot (2 + \mathbb{E}[X])
\end{aligned}
$$


Now we turn to $\mathbb{E}[X \| B]$, the average number of rolls needed to get two 6's in a row given the first roll was not a 6. In this case, we are back at square 1, having wasted one roll. So we can write 

$$
\begin{aligned}
\mathbb{E}[X | B] &= 1 + \mathbb{E}[X]
\end{aligned}
$$

Plugging all of this back into the original equation, we have:

$$
\begin{aligned}
\mathbb{E}[X] &= P(A)\cdot\mathbb{E}[X | A] + P(\text{B})\cdot\mathbb{E}[X | B] \\ 
&= \frac{1}{6} \cdot \bigg[\frac{1}{6}\cdot(2) + \frac{5}{6}\cdot(2 + \mathbb{E}[X])\bigg] + \frac{5}{6}\cdot \bigg[1 + \mathbb{E}[X]\bigg]
\end{aligned}
$$

Solving this equation is simple, we get $\mathbb{E}[X] = 42$. So on average, it takes 42 rolls to get two 6's in a row.

#### A Faster Method
A faster way to arrive at the answer it to consider costs for every roll. Define 3 states:

1.  $S_0$: The beginning state, we have not rolled our first 6 yet.
2. $S_1$: The intermediate state. We have rolled our first 6.
3. $S_2$: The goal state. We have rolled two sixes in a row.

Let $C_i$ denote the "expected cost", or the expected number of rolls to get two sixes in a row when we start in state $i$. Note that this implies $C_2 = 0$, since we do not need to roll any more if we start in the goal state.


The expected cost from $S_0$ can then be written as:

$$
C_0 = \underbrace{\frac{1}{6}}_{\substack{\text{prob of rolling 6} \\ \text{(transition from $S_0$ to $S_1$})}}\cdot(\underbrace{1}_{\text{cost of roll}} + \underbrace{C_1}_{\text{expected cost from $S_1$}}) + \underbrace{\frac{5}{6}}_{\substack{\text{prob of not rolling 6} \\ \text{(transition from $S_0$ to $S_0$)}}}\cdot(\underbrace{1}_{\text{cost of roll}} + \underbrace{C_0}_{\text{expected cost from $S_0$}})
$$

We can break down the expected cost from $S_1$ similarly:

$$
C_1 = \underbrace{\frac{1}{6}}_{\substack{\text{prob of rolling 6} \\ \text{(transition from $S_1$ to $S_2$)}}}\cdot (\underbrace{1}_{\text{cost of roll}} + \underbrace{C_2}_{\text{expected cost from $S_2$}}) + \underbrace{\frac{5}{6}}_{\substack{\text{prob of not rolling a 6} \\ \text{(transition from $S_1$ to $S_0$)}}} \cdot (\underbrace{1}_{\text{cost of roll}} + \underbrace{C_0}_{\text{expected cost from $S_0$}})
$$

We can plug this cost $C_1$ into the expression for $C_0$ to obtain:

$$
C_0 = \frac{1}{6}\cdot \bigg[1 + \underbrace{\frac{1}{6} + \frac{5}{6}\cdot (1 + C_0)}_{C_1}\bigg] + \frac{5}{6}\cdot (1 + C_0)
$$

This equation can be solved by rearranging, yielding $C_0 = 42$. Thus, the expected number of rolls needed to yield two 6's in a row is 42. 

### 6 Followed by 5

We break down the problem similarly. Note that in this case, $\mathbb{E}[X \|B]$ is the same as before since if we don't roll a 6 on the first roll, we are back at square 1 having wasted 1 roll. However, $\mathbb{E}[X \|A]$ $\textit{does}$ change. To see this, consider the possibilities for the second roll given the fact that the previous roll was a 6:

- Event $C$: the next roll is a 5 and the game ends. This happens with probability $\frac{1}{6}$ and we use 1 roll for each event. Thus, we have $P(C) = \frac{1}{6}$ and $\mathbb{E}[X \|C, A] = 2$.
- Event $D$: the next roll is a 6. We are in the "halfway" state again, having wasted 1 roll on this event. We have not been reset to the beginning but we also haven't won. This happens with  probability $\frac{1}{6}$. Thus, we have $P(D) = \frac{1}{6}$ and $\mathbb{E}[X \| D, A] = 1 + \mathbb{E}[X \| A]$.
- Event $E$: the next roll is one of 1 through 4, and we get reset back to square 1 having wasted 1 roll on this event and 1 roll on event $A$. This happens with probability $\frac{4}{6}$. Thus we have $P(E) = \frac{4}{6}$ and $\mathbb{E}[X \|E, A] = 2 + \mathbb{E}[X]$.

Using the law of total expectation again, we have:

$$
\begin{aligned}
\mathbb{E}[X | A] &= P(\text{C})\cdot\mathbb{E}[X |C, A] + P(D)\cdot\mathbb{E}[X | D,A] + P(E) \cdot \mathbb{E}[X | E, A] \\ 
&= \frac{1}{6}\cdot (2) + \frac{1}{6}\cdot(1 + \mathbb{E}[X |A]) + \frac{4}{6}\cdot (2 + \mathbb{E}[X])
\end{aligned}
$$

Rearranging, we find

$$
\begin{aligned}
\mathbb{E}[X | A] 
&= \frac{3}{5} + \frac{4}{5}\cdot (2 + \mathbb{E}[X])
\end{aligned}
$$



Returning to our expression for $\mathbb{E}[X]$ and plugging everything in:

$$
\begin{aligned}
\mathbb{E}[X] &= P(A)\cdot\mathbb{E}[X | A] + P(\text{B})\cdot\mathbb{E}[X | B] \\ 
&= \frac{1}{6} \cdot \bigg[\frac{3}{5} + \frac{4}{5}\cdot (2 + \mathbb{E}[X])\bigg] + \frac{5}{6}\cdot \bigg[1 + \mathbb{E}[X]\bigg]
\end{aligned}
$$

We can solve this equation easily to find that $\mathbb{E}[X] = 36$. So, on average it takes 36 rolls to roll a 6 followed by a 5. Interestingly it is less than rolling two 6's in a row. This is because upon rolling the first 6, there is one additional event for your second roll that does not reset you back to the beginning (rolling another 6). This brings the average number of rolls down.


## Answer via Markov Chains

### Two 6's in a Row


### 6 Followed by 5