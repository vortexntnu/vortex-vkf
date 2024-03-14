# Models
This folder contains the models used in the Vortex-VKF project. The models are derived from either DynamicModelI or the SensorModelI interface. 
They define the dynamics of the system and the sensor models used in the project.

All classes and functions are under the namespace `vortex::models`.
The interfaces that the different models are derived from are definged under the namespace `vortex::models::interface`.

## Overview
- [`dynamic_model_interfaces.hpp`](dynamic_model_interfaces.hpp) contains the interfaces for the dynamic models.
- [`sensor_model_interfaces.hpp`](sensor_model_interfaces.hpp) contains the interfaces for the sensor models.
- [`dynamic_models.hpp`](dynamic_models.hpp) contains some movement models that are commonly used in target tracking.
- [`sensor_models.hpp`](sensor_models.hpp) contains some sensor models that are commonly used.
- [`imm_model.hpp`](imm_model.hpp) contains the IMM model.



## Dynamic Models
Models for describing the dynamics of the system. 

### Dynamic Model Interfaces
The dynamic model interfaces are virtual template classes that makes it convenient to define your own dynamic models. 


#### DynamicModel
Dynamic model interface for other classes to derive from. The [UKF](../filters/README.md#UKF) works on models derived from this class.

##### Key Features
- Static Constants for Dimensions: The base class defines static constants (N_DIM_x, N_DIM_u, N_DIM_v) for dimensions, allowing derived classes to reference these dimensions.
- Type Definitions: It uses Eigen library types for vectors and matrices to represent states, inputs, and noise.
- Pure Virtual Functions: Has the pure virtual functions `f_d` (discrete time dynamics) and `Q_d` (discrete time process noise), enforcing derived classes to implement these functions.
- Sampling Methods: Provides methods for sampling from the discrete time dynamics.

#### DynamicModelLTV
Dynamic model interface for other classes to derive from. The [EKF](../filters/README.md#ekf) (and UKF) works on models derived from this class.

This interface inherits from the `DynamicModel` interface and defines the dynamics of the system as a linear time varying system. The virtual method `f_d` from the `DynamicModel` interface is implemented as a linear time varying system on the form 
$$
f_d(dt, x_k, u_k, v_k) = x_{k+1} = A_d(dt, x_k) x_k + B_d(dt, x_k) u_k + G_d(dt, x_k)v_k
$$
where $dt$ is the time step, $x_k$ is the state at time $k$, $u_k$ is the input at time $k$ and $v_k$ is the process noise at time $k$. The matrices $A_d$, $B_d$ and $G_d$ are defined as virtual methods and must be implemented by the derived class.

##### Usage
In order to define a *new* dynamic model, the user must first create a new class that inherits from the `DynamicModelLTV` interface. The user must then override the methods `A_d`, `B_d`, `G_d` and `Q_d` for the discrete time dynamics.

```cpp
#include <vortex_filtering/vortex_filtering.hpp>

class MyDynamicModel : public interface::DynamicModelLTV<N_DIM_x, N_DIM_u, N_DIM_v> {
public:
    // Get all types used in the models
    using T = vortex::Types_xuv<N_DIM_x, N_DIM_u, N_DIM_v>;


    // Define the matrices A_d, (B_d), (G_d) and Q_d
    T::Mat_XX A_d(double dt, const T::Vec& x) const override {
        // Implement the A_d matrix
    }

    T::Mat_XU B_d(double dt, const T::Vec& x) const override {
        // Implement the B_d matrix
    }

    // ... and so on
};
```


#### DynamicModelCTLTV
This class implements the `DynamicModelLTV` interface and defines the dynamics of the system as a continuous-time linear time-varying system. The matrices `A_c`, `B_c` `Q_d` and `G_c` are defined as virtual methods and must be implemented by the derived class. The matrices `A_d`, `B_d`, `Q_d` and `G_d` are then generated using [exact discretization](https://en.wikipedia.org/wiki/Discretization).

### IMM Model
__Interacting Multiple Models__
This class can store multiple `DynamicModel` objects and defines functions to calculate the probability of switching between the models. 

#### Usage
To instantiate a **Interacting Multiple Models (IMM) object**, you must provide four parameters:

1. **Hold Times Vector**: This vector should contain the expected time durations that each model should be held before a switch occurs. The length of this vector must equal the number of models `N`. _When the switch occurs_ is modeled by an exponential distribution with parameters given in the hold times vector.

2. **Switching Probabilities Matrix**: This is an `N x N` matrix where each element at index `(i, j)` represents the probability of transitioning from model `i` to model `j`. Each probability lies between 0 and 1, and the sum of probabilities in each row equals 1. These probabilities define the likelihood of transitioning to a particular model given that a switch occurs. The diagonal should be zero as this represents the probability of _switching to itself_, which doesn't make sense.

3. **Dynamic Model Objects**: A list of `DynamicModel` objects, one for each model in use.

4. **State Names**: An array of state names (enums) for each state in each model. This is used to compare the states of the different models with each other in the [IMM filter](../filters/README.md#imm-filter) for proper mixing of the states.

#### Example
```cpp
// Create aliases for the dynamic models (optional)
using CP = vortex::models::ConstantPosition;
using CV = vortex::models::ConstantVelocity;
using CT = vortex::models::CoordinatedTurn;

// Create alias for the IMM model (optional, but probably a good idea)
using IMM = vortex::models::IMMModel<CP, CV, CT>;

// Specify holding times and switching probabilities
Eigen::Vector3d hold_times{1.0, 2.0, 3.0};
Eigen::Matrix3d switch_probs{
    {0.0, 0.5, 0.5},
    {0.5, 0.0, 0.5},
    {0.5, 0.5, 0.0}
};

double std_pos = 0.1, std_vel = 0.1, std_turn = 0.1;

// Specify the state names of the models
using ST = vortex::models::StateType;
std::array<ST, 2> cp_names{ST::pos, ST::pos};
std::array<ST, 4> cv_names{ST::pos, ST::pos, ST::vel, ST::vel};
std::array<ST, 5> ct_names{ST::pos, ST::pos, ST::vel, ST::vel, ST::turn};

/* Note: for the models in this example, you can use the already defined state names instead:
    auto cp_names = CP::StateNames;
    auto cv_names = CV::StateNames;
    auto ct_names = CT::StateNames;
But for custom models, you will have to define the state names yourself.
*/

// initialize IMM with the hold times, switching probabilities, dynamic models and state names
IMM imm_model(hold_times, switch_probs, 
              {CP(std_pos), cp_names}, 
              {CV(std_vel), cv_names}, 
              {CT(std_vel, std_turn), ct_names});

// Enjoy your very own IMM model! :)
```

#### Theory
It's important to note that the actual probability of switching from one model to another is determined through the `hold_times` vector. By treating the system as a **Continuous Time Markov Chain (CTMC)**, as detailed on [Wikipedia](https://en.wikipedia.org/wiki/Continuous-time_Markov_chain), the model calculates the switching probabilities based on the specified hold times and the switching probabilities matrix. 



### Dynamic Models
`dynamic_models.hpp` 
This file contains some movement models that are commonly used in an IMM.
- `ConstantVelocity`: Has states for position and velocity. The template parameter `n_spatial_dims` specifies the number of spatial dimensions. So if the model is used in 2D, `n_spatial_dims` should be set to 2 and the model will have 4 states. `x`, `y`, `v_x` and `v_y`.
- `ConstantAcceleration`: Has states for position, velocity and acceleration. The template parameter `n_spatial_dims` specifies the number of spatial dimensions. So if the model is used in 2D, `n_spatial_dims` should be set to 2 and the model will have 6 states. `x`, `y`, `v_x`, `v_y`, `a_x` and `a_y`. 
- `CoordinatedTurn`: Has states for 2D position, 2D velocity and turn rate. 


## Sensor Models
### Sensor Model Interfaces

#### SensorModel
This interface defines the sensor models. The methods `h` `H` and `R` define the sensor model. The method `h` is the measurement function, `H` is the Jacobian of the measurement function and `R` is the measurement noise covariance matrix. The sensor model is assumed to be defined in discrete time. The noise is assumed additive and Gaussian.

##### Usage
In order to define a new sensor model, the user must create a new class that inherits from the SensorModelI interface. The user must then implement the methods `h`, `H` and `R` as they are pure virtual.

The interface is a template class with parameter `N_DIM_x` and `N_DIM_z`. `N_DIM_x` is the dimension of the state vector and `N_DIM_z` is the dimension of the measurement vector. The user must specify these when creating a new class, or derive a template class from the SensorModelI interface.


