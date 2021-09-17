# Kernels


## Kernel trick
&emsp;When $y$ can be more accurately represented as a non-linear function of $x$, a cubic function of the variable $x$ can be viewed as a linear function over the variables $\phi(x)$.
$$
    \phi(x) = \begin{bmatrix}
        1 \\
        x \\
        x^{2} \\
        x^{3}
    \end{bmatrix} \in \mathbb{R}^{4} \\
    \theta_{3}x + \theta_{2}x + \theta_{1}x + \theta_{0} = \theta^{T}\phi(x)
$$
&emsp;We call $\phi$ a **feature map**,which maps the features from low-dimensional to high-dimensional.
&emsp;However, The vector $\theta$ itself is of high dimension. We must take a long time to update every entry of $\theta$ and store it. Thus, we will introduce the **kernel trick** with which we will not need to store θ explicitly, and the runtime can be significantly improved.
$$
\theta^{T}\phi(x) = \sum_{i=1}^{n}\beta_{i}\phi(x(i))^{T}\phi(x) = \sum_{i=1}^{n}\beta_{i}\mathit{K}(x^{(i)},x)
$$

## Properties of kernels
&emsp;If $\mathit{K}$ is a valid kernel,then $\mathit{K}_{ij} = \mathit{K}(x^{(i)},x^{(j)}) = \phi(x^{(i)})^{T}\phi(x^{(j)}) = \phi(x^{(j)})^{T}\phi(x^{(i)}) = \mathit{K}(x^{(j)},x^{(i)}) = \mathit{K}_{ji}$, and hence $\mathit{K}$ must be symmetric.
$$
z^{T}\mathit{K}z = \sum_{i}\sum_{j}z_{i}\mathit{K}_{ij}z_{j} = \sum_{k}(\sum_{i}z_{i}\phi_{k}(x^{(i)}))^{2} \geq 0
$$
&emsp;Since $z$ was arbitrary, this shows that $\mathit{K}$ is positive semi-definite($\mathit{K} \geq 0$). Hence, we’ve shown that if $\mathit{K}$ is a valid kernel (i.e., if it corresponds to
some feature mapping $\phi$), then the corresponding kernel matrix $\mathit{K} \in \mathbb{R}^{n\times n}$ is symmetric positive semidefinite.