# Filters
This folder contains the filters. They are based on Kalman filters

All classes and functions are under the namespace `vortex::filters`.

## EKF
This class represents an [Extended Kalman Filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter). It is a template class with parameters `DynamicModelT` and `SensorModelT`. `DynamicModelT` is the dynamic model used in the filter and `SensorModelT` is the sensor model used in the filter. There is no need to specify the dimensions of the dynamic model or the sensor model as they are automatically retrieved from the models.

### Usage
To create an instance you need to provide a dynamic model and a sensor model. The dynamic model must be derived from the DynamicModelI interface and the sensor model must be derived from the SensorModelI interface. The EKF object will then store a copy of the dynamic model and the sensor model. (It should probably store a const reference instead to enable external live tuning for the parameters though). It does not store state estimates or covariance matrices.

## IMM Filter
This class represents an [Interacting Multiple Model Filter](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/14-Adaptive-Filtering.ipynb). It is a template class with parameters `DynamicModelT` and `SensorModelT`. `DynamicModelT` is the dynamic model used in the filter and `SensorModelT` is the sensor model used in the filter.