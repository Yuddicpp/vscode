# Support Vector Machines

&emsp;This note will present the Support Vector Machine(SVM) learning algorithm. SVMs are among the best (and many believe are indeed the best) “off-the-shelf” supervised learning algorithms.

## Functional and geometric margins
&emsp;We will be considering a linear classifier for a binary classification problem with labels y and features x. From now, we’ll use $y \in \{-1,1\}$(instead of 0,1) to denote the class labels. Also, rather than parameterizing our linear classifier with the vector θ, we will use parameters w, b, and write our classifier as:
$$
    h_{w,b}(x) = g(w^{T}x+b)
$$
Here, $g(z) = 1$ if $z\geq0$, and $g(z) = -1$ otherwise. $b$ takes the role of what was previously $\theta_{0}$, and w takes the role of $[θ_{1} . . . θ_{d}]^{T}$.
&emsp;Define the **functional margin** of $(w,b)$ with respect to the training examples as:
$$
    \hat{\gamma}^{(i)} = y^{(i)}(w^{T}x^{(i)}+b)
$$
If $y^{(i)} = 1$, then the functional margin will be very large. Conversely, if $y^{(i)} = -1$, the functional margin will also be very large. Moreover, if $y^{(i)}(w^{T}x^{(i)}+b) > 0$, our prediction on this example is correct.
&emsp;The geometric margin is defined as
$$
    \gamma^{(i)} = y^{(i)}((\frac{w}{||w||})^{T}x^{(i)}+\frac{b}{||w||}) = \frac{\hat{\gamma}^{(i)}}{||w||}
$$
At the same time, we also define the geometric margin of (w, b) to be the smallest of the geometric margins on the individual training examples:
$$
    \gamma = \min_{i=1,...,n}\gamma^{(i)}
$$

## The optimal margin classifier
&emsp;We will introduce the scaling constraint that the functional margin of w, b with respect to the training set must be 1:
$$
    \hat{\gamma}=1.
$$
Since multiplying w and b by some constant results in the functional margin being multiplied by that same constant, this is indeed a scaling constraint, and can be satisfied by rescaling w, b.Noting that maximizing $\hat{\gamma}/||w|| = 1/||w||$  is the same thing as minmizing$||w||^{2}$, we now have the following optimization problem:
$$
\begin{aligned}
    \min_{w,b} & \frac{1}{2}||w||^{2} \\
    s.t. & y^{(i)}(w^{T}x^{(i)}+b) \geq1,i =1,...,n
\end{aligned}
$$