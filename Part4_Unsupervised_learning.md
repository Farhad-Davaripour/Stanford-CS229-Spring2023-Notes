# Part 4: Unsupervised Learning
Unsupervised learning is employed to manipulate and derive insight from unlabelled data.
### K-means
The K-means algorithm is a widely used clustering method for grouping unlabeled data into a specified number of clusters. It operates through iterative steps until convergence is reached. These steps involve:

- Assuming the value of k, representing the desired number of clusters.
- Randomly initializing centroid positions for each cluster.
- Assigning each data point to the nearest cluster centroid.
- Updating the centroid positions and assignments.  

Convergence is determined by achieving a distortion threshold, which measures the sum of squared distances between data points and their assigned cluster centroids. To optimize the assignments and centroid positions, the algorithm employs coordinate descent. In each iteration, it minimizes the distortion function with respect to one constraint (either centroid coordinates or point assignments) while treating the other as constant. Since the distortion function is not a convex function (i.e., there are multiple local minima), once the optimization is converged, it will be repeated by new initialization of centroids. Finally, the iteration, which leads to lowest distortion value, is selected for grouping the data points.

### Mixture of Gaussian
"The mixture of Gaussian model is an unsupervised algorithm used for clustering data into a certain number of groups, where each group is represented by a Gaussian distribution. The goal is to estimate the optimal means and covariances of the Gaussian distributions for each group and uncover the assignment of each data point to the correct group, which is encoded in a hidden variable called the `latent variable`. 

To train the model and find the optimal parameters and assignments, the `EM algorithm` is employed. The algorithm iteratively improves the estimation by performing the E-step and M-step. In the E-step, the latent variable is approximated to determine the probabilities of data points belonging to each group. In the M-step, the distribution parameters are updated based on the  assignments obtained in the E-step. The refinement of assignments and parameter estimates continues until convergence.

In simpler terms, in the E-step (Expectation), we assign each data point a "weight" for each Gaussian distribution in the mixture, reflecting the probability of that data point being generated from that Gaussian. These weights indicate how similar a data point is to each Gaussian's current parameter values. Then, in the M-step (Maximization), we adjust the parameters of each Gaussian distribution (the mean and variance), as well as their proportions in the mixture, to maximize the likelihood of the observed data given these parameters. This is effectively using our weights from the E-step to update our guesses for the parameters in a way that better matches the data. These two steps are iteratively repeated until the parameters and weights stop changing significantly, indicating that the best fit for the data has been found.

### Jensen's inequality
Jensen's inequality is a mathematical concept that enables us to monitor the behavior of a function which in the context of EM algorithm helps us to ensure the likelihood of the observed data is consistently increasing. The application of Jensen's inequality on convex curve states that for any two variable values a and b, the the mid point of of the application of function f to points a and b separately $E(fx)$ is greater or equal to the application of function f to the midpoint between a and b $f(E(x))$. This is also visualized in the figure below:

<p align="center">
  <img src="Figure/Jensens_inequality.png" alt="Hypothesis Function" width="400"/>
</p>

### Principal Component Analysis (PCA)

PCA is an efficient method to reduce data dimensionality by maintaining the majority of variance within the data. It is done by normalizing the data and then identifying the direction in the data space that has the highest variation of data, which is called `major axis of variation` or the first principal component. Other components are derived the same way by with a new constraint that each subsequent component should be orthogonal to the previous component to avoid correlation between components. Hence, using PCA the data with original n features could now be represented by fewer number of components. This technique has multiple use cases in data compression (e.g., representing images with a small number of components), data visualization (e.g., visualizing high dimensional data in a 2d space), data pre-processing (e.g., to reduce computation and overcome overfitting), and noise reduction.  

It's worth noting that each component in PCA is a combination of the original data features. A high coefficient of a feature within a component indicates that this feature exhibits high variability, but it does not necessarily imply that the feature is important in the context of predictive modeling. Feature importance for prediction models is generally measured by assessing the impact of each feature on the model's predictive performance.

### Independent Component Analysis (ICA)

ICA is a technique to separate source data from mixed data (e.g., overlapped images). A typical use case of ICA is to separate the voice of each speaker in a cocktail party from the recordings taken at random locations within the party venue. The ICA could be mathematically represented by x = As where x is the observation (mixed data), s is the source data, and A is the unknown mixing matrix. The goal is to find the unmixing matrix $A^{-1}$ (denoted as W), which, when applied to the mixed data x, allows us to recover the original source data: $s = Wx$.

Here is the process to recover the source data: 
- The Cumulative Distribution Function (CDF) of each source is assumed to be a sigmoid function.
- The Probability Density Function (PDF) of each source is then obtained as the derivative of the CDF.
- Assuming the sources are independent, the joint PDF of all sources is expressed as a product of the individual source PDFs.
- The PDF of the mixed data is then calculated as a transformed version of the joint PDF, where the transformation is dictated by the mixing matrix.
- The likelihood function is then formed as the product of the values of this PDF evaluated at the observed data points.
- This likelihood function is maximized to estimate the unmixing matrix, W.
- Finally, the estimated W is used to recover the original sources from the mixed data.

### Self supervised learning and foundation models
`Foundation models` are a class of models that are initially trained on substantial volumes of unlabeled data and can be fine tuned with a smaller quantity of labeled data to execute specific tasks. These models utilize `self-supervised learning`, a variant of unsupervised learning, where parts of the input data serve as self-generated labels, thus reducing the need for human intervention in the labelling process. Additionally, foundation models leverage the power of `transfer learning` and deep learning methodologies to uncover the underlying structures within the data. Here, transfer learning refers to the technique of using a pre-trained model on a new but related task, with only minimal additional training.

The foundation models paradigm consists of two stages: `pretraining` on a large, unlabeled dataset to learn general patterns in data, and then `adaptation` to a specific task, often involving a smaller, labeled dataset. During the adaptation phase, two main methods are used: the `linear probe` and `finetuning`. In the linear probe method, a linear layer (also called a "head") is added on top of the pretrained model. The weights of this new layer are then learned from the labeled data of the task, while the rest of the model remains unchanged. The method is particularly useful when working with a small labeled dataset, as it reduces the risk of overfitting by limiting the number of parameters that are learned from this data. On the other hand, finetuning adjusts not only the weights of the added layer but also the weights of the pretrained model itself. This requires more labeled data as it modifies more parameters and could lead to overfitting if not properly regulated.

Also, in addition to fine-tuning and linear-probe, there are two other important methods for model adaptation: `zero-shot learning` and `in-context learning`. In zero-shot learning, there are no specific training data for the task at hand, so we rely on the knowledge that the model has learned during its pretraining phase to generate answers. This method leverages the broad knowledge that models like GPT-3 have accumulated during training on a large amount of text. On the other hand, in-context learning involves providing the model with a few examples of the task at hand, directly in the input prompt. The model uses these examples as a guide to perform the task, effectively learning from the context provided in the prompt. 

#### Computer vision

The pretraining in `computer vision` involves two primary methods: `supervised pretraining` and `contrastive pretraining`. In supervised pretraining, a large, labeled dataset is used to train a neural network, and the output layer is removed afterward, leaving the model's internal `representation` intact for further tasks. On the other hand, contrastive learning identifies and leverages contrasts within the data. It ensures that similar images map to similar representations, while dissimilar images map to distinct representations. Then, during further training, the model aims to bring the representations of similar images closer together while pushing dissimilar ones apart.

#### Natural Language Processing (NLP)
As for NLP, the pretraining process involves converting the words into numerical representations, called `embeddings`, which capture the `semantic` relationships among words. These embeddings are then passed through a `Transformer` model that computes the probability distribution for each possible subsequent word based on the input sequence and outputs the one with highest probability. The `softmax` function is used to normalize these probabilities so the sum is 1. Lastly, the model's parameters are optimized by minimizing the `negative log likelihood`, a measure of difference between the model's predictions and the actual values. This training process adjusts the Transformer model to produce predictions that are as close as possible to the true subsequent words in the input sequences. The transformer model is a neural network specifically designed to handle sequence data.
