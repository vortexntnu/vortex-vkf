# Probability
This folder contains the probability distributions used.

All classes and functions are under the namespace `vortex::prob`.

## MultiVarGauss
This class represents a multivariate Gaussian distribution. It is a template class with parameter `N_DIM`. This is the dimension of the distribution.

### Usage
The class contains methods for getting the mean and covariance, sampling from the distribution, calculating the probability density function and the log probability density function. It works with both static and dynamic dimensions.

To create an instance you need to provide a mean vector and a covariance matrix. The covariance matrix must be symmetric and [positive definite](https://en.wikipedia.org/wiki/Definite_matrix). The mean vector must have the same dimension as the covariance matrix and the MultiVarGauss object.

## GuassianMixture
This class represents a Gaussian mixture distribution. It is a template class with parameter `n_dims`. This is the dimension of the distribution.

### Usage
The class contains methods for sampling from the distribution, calculating the probability density function and the log probability density function. But most importantly the mean and covariance of the distribution can be retrieved. It works with both static and dynamic dimensions.

To create an instance you need to provide a vector of weights and a vector of MultiVarGauss objects. MultiVarGauss objects must have the same dimension as the GaussianMixture object.

To reduce the gaussian mixture to a single gaussian, the methods `GaussianMixture::reduce` and `GaussianMixture::ml_estimate` can be used. The first method reduces the mixture to a single gaussian by matching the mean and covariance of the mixture to a single gaussian. The second method returns the gaussian component with the highest weight multiplied by the mean.
