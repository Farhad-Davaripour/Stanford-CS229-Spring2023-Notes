# Part 4: Unsupervised Learning
Unsupervised learning is employed to manipulate and derive insight from unlabelled data.
### K-means
The K-means algorithm is a widely used clustering method for grouping unlabeled data into a specified number of clusters. It operates through iterative steps until convergence is reached. These steps involve:

- Assuming the value of k, representing the desired number of clusters.
- Randomly initializing centroid positions for each cluster.
- Assigning each data point to the nearest cluster centroid.
- Updating the centroid positions and assignments.  

Convergence is determined by achieving a distortion threshold, which measures the sum of squared distances between data points and their assigned cluster centroids. To optimize the assignments and centroid positions, the algorithm employs coordinate descent. In each iteration, it minimizes the distortion function with respect to one constraint (either centroid coordinates or point assignments) while treating the other as constant. Since the distortion function is not a convex function (i.e., there are multiple local minima), once the optimization is converged, it will be repeated by new initialization of centroids and the iteration which leads to lowest distortion value is selected for grouping the data points.