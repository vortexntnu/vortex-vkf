# Filters
This folder contains various filters

All classes and functions are under the namespace `vortex::filters`.

## EKF
This class represents an [Extended Kalman Filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter). It is a template class with parameters `DynamicModelT` and `SensorModelT` for the dynamic model and sensor model respectively. The models have to be derived from `vortex::models::DynamicModelLTV` and `vortex::models::SensorModelLTV`. All methods are static, so there is no need to create an instance of this class.

### Usage
```cpp
// Make typedef for the EKF using the dynamic and sensor models
using EKF = vortex::filters::EKF<DynamicModelT, SensorModelT>;

// Create the dynamic model and sensor model
auto dynamic_model = std::make_shared<DynamicModelT>(...);
auto sensor_model = std::make_shared<SensorModelT>(...);

// Initial values
Gauss_x x_est_prev = ...;
Vec_z z_meas = ...;

// Estimate the next state
auto [x_est_upd, x_est_pred, z_est_pred] = EKF::step(dynamic_model, sensor_model, dt, x_est_prev, z_meas);
```

## UKF

[UKF explained](https://towardsdatascience.com/the-unscented-kalman-filter-anything-ekf-can-do-i-can-do-it-better-ce7c773cf88d)

The UKF can take any model derived from `vortex::models::DynamicModel` and `vortex::models::SensorModel`. All methods are static, so there is no need to create an instance of this class.

### Usage
```cpp
// Same as EKF
```

## IMM Filter
This class represents an [Interacting Multiple Model Filter](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/14-Adaptive-Filtering.ipynb). It is a template class with parameters `SensModelT` and `ImmModelT` for the sensor model and IMM model respectively. All methods are static, so there is no need to create an instance of this class.