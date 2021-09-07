# Lecture 5 - Generative Learning algorithms & Naive Bayes
&emsp;Algorithms that try to learn p(y|x) **directly** (such as logistic regression),or algorithms that try to learn mappings **directly** from the space of inputs X to the labels {0, 1}, (such as the perceptron algorithm) are called **discriminative learning algorithms**.
&emsp;Algorithms that instead try to model p(x|y) (and p(y)) are called **generative learning algorithms**.

## Gaussian discriminant analysis

### The multivariate normal distribution(Gaussian distribution)
$$ p(x;\mu,\Sigma) =\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} exp(-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)) $$
&emsp;In this equation, $\mu \in \mathbb{R}^{d}$ is a **mean vector**, and $\Sigma \in \mathbb{R}^{d \times d}$ is a **covariance matrix**. $\Sigma \geq 0$ is symmetric and positive semi-definite. This equation is also written $\mathcal{N}(\mu, \Sigma)$
&emsp;For a random variable $X$ distributed $\mathcal{N}(\mu, \Sigma)$, the mean is given by $\mu$:
$$E[X] = \int_{x} xp(x;\mu,\Sigma)dx = \mu$$
&emsp;The covariance of a vector-valued random variable $X$ is given by $\Sigma$:
$$
\begin{aligned}
    Cov(X) &= E[(X-E[X])(X-E[X])^{T}] \\
    &= E[XX^{T}]-(E[X])(E[X])^{T} \\
    &= \Sigma
\end{aligned}
$$

### The Gaussian Discriminant Analysis model
&emsp;The model is:
$$
\begin{aligned}
    y &\sim Bernoulli(\phi) \\
    x|y=0 &\sim \mathcal{N}(\mu _{0}, \Sigma) \\
    x|y=1 &\sim \mathcal{N}(\mu _{1}, \Sigma)
\end{aligned}
$$
Writing out the distributions, this is:
$$
\begin{aligned}
    p(y) &= \phi^{y}(1-\phi)^{1-y} \\
    p(x|y=0) &= \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} exp(-\frac{1}{2}(x-\mu _{0})^{T}\Sigma^{-1}(x-\mu _{0})) \\
    p(x|y=1) &= \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} exp(-\frac{1}{2}(x-\mu _{1})^{T}\Sigma^{-1}(x-\mu _{1}))
\end{aligned}
$$
&emsp;This model is usually applied using only **one covariance matrix** $\Sigma$. The log-likelihood of the data is given by:
$$
\begin{aligned}
    \mathcal{l}(\phi,\mu_{0},\mu _{1},\Sigma) &= \log \prod_{i=1}^{n}p(x^{(i)},y^{(i)};\phi,\mu _{0},\mu _{1},\Sigma) \\
    &=  \log \prod_{i=1}^{n}p(x^{(i)}|y^{(i)};\phi,\mu _{0},\mu _{1},\Sigma)p(y^{(i);\phi})
\end{aligned}
$$
By maximizing $\mathcal{l}$ with respect to the parameters, we find the maximum likelihood estimate of the parameters to be:
$$
\begin{aligned}
    \phi &= \frac{1}{n} \sum_{i=1}^{n} 1 \{y^{(i)}=1\} \\
    \mu _{0} &= \frac{\sum^{n}_{i=1} 1 \{ y^{(i)} =0 \} x^{(i)} }{\sum^{n}_{i=1} 1 \{ y^{(i)} =0 \}} \\
    \mu _{1} &= \frac{\sum^{n}_{i=1} 1 \{ y^{(i)} =1 \} x^{(i)} }{\sum^{n}_{i=1} 1 \{ y^{(i)} =1 \}} \\
    \Sigma &= \frac{1}{n} \sum^{n}_{i=1} (x^{(i)} - \mu _{y^{(i)}})(x^{(i)} - \mu _{y^{(i)}})^{T}
\end{aligned}
$$

## Naive Bayes