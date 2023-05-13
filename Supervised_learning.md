# Part 1: Supervised Learning

In supervised learning, the goal is to train a  hypothesis function h(x) to map the input x to the output y (h:X→Y).

<p align="center">
  <img src="Figure/Hypothesis_function.png" alt="Hypothesis Function" width="250"/>
</p>

---

## Chapter 1: Linear Regression: 
The hypothesis function in linear regression for a two dimentional input (a regression problem with two features: x<sub>1</sub> and x<sub>2</sub> ) could be represented as below:

$h_{\theta}(x)/h(x) = \theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} = \theta^{T}x$, where:

- $\theta_{0}$ is the y-intercept or bias term.
- $\theta_{1}$ and $\theta_{2}$ are the weights for features $x_{1}$ and $x_{2}$, respectively.
- $\theta^{T}$ is the transpose of the weight vector, represented as a row vector.
- $x$ is the input feature vector, represented as a column vector.

The next step is to forulate a loss function to represent the discreapancy or error of $h(x)$. The most common closs function for a linear regression algorithm is the sum of mean squared erro (or Least squares cost function):

$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2$

`Note` that the mean squread erorr loss function is a convex function since it has a unique global minimum.

### Least Mean Square (LMS)
Having the loss function, the next step is to minimize the loss function through an iterative process (e.g., using gradient decsent algorithm):

$\theta_{j} := \theta_{j} - \alpha \frac{\partial}{\partial \theta_{j}} J(\theta)$

where:
- $\theta_{j}$ is the j-th weight parameter.
- $\alpha$ is the learning rate that controls the step size in gradient descent.
- $\frac{\partial}{\partial \theta_{j}} J(\theta)$ is the partial derivative of cost function with respect to $\theta_{j}$ used to update the parameter.

`Note` that each iteration in Gradient Descent can incorporate either one random training sample or the entire samples. When using the entire samples in each step, it's called Batch Gradient Descent (BGD) which is typically just referred to as Gradient Descent. In the former case, where only one random training sample is used, it's called Stochastic Gradient Descent (SGD).

`Note` that the weights of the hypothesis function could be determined directly using the below close form analytical equation (termed normal equation) which is more suitable for small to medium sized dataset compared to the numerical optimization approaches (e.g., gradient descent):  

$θ = (X^T X)^{-1} X^T y~$  
where:
- $\theta$ = vector of the optimal parameter values
- $X$ = matrix of the training examples
- $\tilde{y}$ = vector of target values

---
## Chapter 2: Classification and logistic regression
Classification is quite similar to the regression except here the output only includes a limited number of discrete values, for instance if it is a binary classification, then the output is either 0 or 1 where 1 is considered the positive class and 0 the negative class.
 
 ### Logistic regression
In a classification problem (e.g., a binary classification), the hypothesis function should output limited discrete values. Therefore, using a linear regression hypothesis function may not work effectively, and instead, a logistic or sigmoid function is used:

$g(z) = 1 / (1 + e^{(-z)})$

$g(θᵀx) = hθ(x) = 1 / (1 + e^{(-θᵀx)})$ 

where $hθ(x)$ is bounded between 0 and 1. 

Similar to linear regression, in order to find the weights for the hypothesis function we should find the corresponding loss function and then minimize it using gradient descent.

The likelihood function which gives the probability of observing 1 as output label given x as training data and θ as the parameter is as below.

$L(θ) = ∏ᵢ p(yᵢ|Xᵢ, θ) = ∏[hθ(x^{(i)})]^{y^{(i)}} * [1 - hθ(x^{(i)})]^{(1-y^{(i)})}$  
where:
- `∏ᵢ` is the product operator that multiplies the probabilities
- `p(yᵢ|Xᵢ, θ)` is the probability of having `yᵢ` given `xᵢ` and `θ`. 

The weight parameters could be obtained by maximizing the likelihood function above, using gradient descent:

$θj := θj + α∂/∂θj(log L(θ))$ 

Alternatively, weight parameters could be obtained by minimizing the loss function which is -log of likelihood function:

$J(θ) = (-1/m) * ∑(i=1 to m) [y^{(i)}*log(hθ(x^{(i)})) + (1-y^{(i)})*log(1-hθ(x^{(i)}))]$

Using loss function, the gradient descent is:  

$θj := θj - α * ∂J(θ) / ∂θj$

`Note` that alternatively weight parameters could be obtained using Newton's method by maximizing the log likelihood function iteratively. In each iteration the Newton's method finds the root of the first derivative of the function (known as `score function`) to get the direction and the second derivative (`Hessian matrix`) to get step size for updating the weight parameter of the hypothesis function and repeats the process until converging to the max of log likelihood function. The Newton's optimization method is used sometimes over gradient descent as it can improve convergence speed in certain cases.

`Note` that if instead of the sigmoid function, a simple threshold function is utilized as the hypothesis function (i.e., g(z) = 1 if z>=0 and g(z)=0 if z<0>), then the algorithm is called the perceptron learning algorithm.

### Multi-class classification

Although the idea of binary classification could be expanded to be used in multi class problems, there are direct methods to train a model and extract the weight parameters. The `multinomial` model is a suitable model which aims at assigning a probability to each possible outcome by meeting the condition that the sum of the probabilities should be 1. It uses a `softmax` function which takes the vector of inputs (`logits`) and transform them into probability vectors. The `softmax` function is then used to compute the probability for every possible outcome. Then the `cross entropy loss` which is a modularized form of `negative log likelihood` is used to compute the discrepancy of the model predictions. The `gradient descent` algorithm is then used to derive the weight parameters.  
---
## Chapter 3: Generalized linear model
Generalized linear models (GLMs) utilize the framework of exponential family distributions to analyze the relationship between independent and dependent variables. In GLMs, the choice of distribution depends on the nature of the problem, with the Bernoulli distribution commonly used for classification tasks and the Gaussian (or normal) distribution for regression.

`Note` that the Gaussian (normal) distribution, synonymous in this context, is a continuous probability distribution characterized by a symmetrical bell-shaped curve. It is typically employed in regression settings to model continuous variables. Whereas, the Bernoulli distribution is a discrete probability distribution designed for binary outcomes. It assigns probabilities to two possible outcomes, typically labeled as 0 and 1, making it suitable for classification problems.

`Note` that the exponential family distribution is a mathematical form of probability distribution. It is termed as such as the exponential term within it's probability density function has the key role. So the exponential distribution which is a specific distribution within the exponential family should not be mistaken with the family distribution itself. 

---
## Chapter 3: Generative learning algorithm
There are two main types of learning algorithms when it comes to classification problems. The generative learning algorithm such as Gaussian Naive Bays, which attempts to model the underlying probability distribution of the input variable for each class separately; and the discriminative learning algorithm such as logistic regression which aims at finding the best decision boundary to separate different classes and directly mapping the input variables to the output class labels.

`Note` that the discriminative learning algorithm uses conditional probability distribution (p(y|x)) which is the probability distribution of y given input features x. On the other hand, the generative learning algorithm uses joint distribution (p(x, y)) which is the probability distribution of both input feature x and output label y. This enables generative learning algorithms to be able to generate synthetic data which replicates the same probability distribution of training data.

`Note` that the joint probability distribution represents the probability of the simultaneous occurrence of all possible combinations of target variables and features. It is denoted as P(X1, X2, ..., Xn), where xi is a random variable. On the other hand, the conditional probability distribution focuses on finding the probability of the target variable given the features. It is represented as P(Y|X) = P(X, Y) / P(X), where P(X, Y) is the joint probability distribution and P(X) is the marginal probability distribution of variable X.