# Models
This folder contains the models used in the Vortex-VKF project. The models are derived from either DynamicModelI or the SensorModelI interface. 
They define the dynamics of the system and the sensor models used in the project.

All classes and functions are under the namespace `vortex::models`.
The interfaces that the different models are derived from are definged under the namespace `vortex::models::interface`.

## Dynamice Models
Models for describing the dynamics of the system. 

### Dynamic Model Interfaces



#### DynamicModel
Dynamic model interface for other classes to derive from. The [UKF class](../filters/README.md#UKF) works on models derived from this class.

##### Key Features
- Static Constants for Dimensions: The base class defines static constants (N_DIM_x, N_DIM_u, N_DIM_v) for dimensions, allowing derived classes to reference these dimensions.
- Type Definitions: It uses Eigen library types for vectors and matrices to represent states, inputs, and noise.
- Pure Virtual Functions: Has the pure virtual functions `f_d` (discrete time dynamics) and `Q_d` (discrete time process noise), enforcing derived classes to implement these functions.
- Sampling Methods: Provides methods for sampling from the discrete time dynamics.

#### DynamicModelLTV
Dynamic model interface for other classes to derive from. The [EKF class](../filters/README.md#ekf) works on models derived from this class.

This interface inherits from the `DynamicModel` interface and defines the dynamics of the system as a linear time varying system. The virtual method `f_d` from the `DynamicModel` interface is implemented as a linear time varying system on the form 
$$
f_d(dt, x_k, u_k, v_k) = x_{k+1} = A_d(dt, x_k) x_k + B_d(dt, x_k) u_k + G_d(dt, x_k)v_k
$$
where $dt$ is the time step, $x_k$ is the state at time $k$, $u_k$ is the input at time $k$ and $v_k$ is the process noise at time $k$. The matrices $A_d$, $B_d$ and $G_d$ are defined as virtual methods and must be implemented by the derived class.

##### Usage
In order to define a *new* dynamic model, the user must first create a new class that inherits from the `DynamicModelLTV` interface. The user must then override the methods `A_d`, `B_d`, `G_d` and `Q_d` for the discrete time dynamics. In addition, the typedef `DynModI` has to be present in the derived class and should point to the `DynamicModelLTV` class like this:

```cpp
class MyDynamicModel : public interface::DynamicModelLTV<N_DIM_x, N_DIM_u, N_DIM_v> {
    // ...
    using DynModI = interface::DynamicModelLTV<N_DIM_x, N_DIM_u, N_DIM_v>;
    // ...
    using typename DynModI::Gauss_x;
    using typename DynModI::Vec_x;
};
```

The purpose of this typedef is to allow convenient access to types like `Gauss_x`, `Vec_x`, etc., without needing to redefine them in the derived class. It also allows other classes to access the types defined in the `DynamicModelLTV` class. For example, the `EKF` class uses this typedef to access the sizes of the state, input and noise vectors inherent to the dynamic models.


#### DynamicModelCTLTV
This class implements the `DynamicModelLTV` interface and defines the dynamics of the system as a continuous-time linear time-varying system. The matrices `A_c`, `B_c` `Q_d` and `G_c` are defined as virtual methods and must be implemented by the derived class. The matrices `A_d`, `B_d`, `Q_d` and `G_d` are then generated using [exact discretization](https://en.wikipedia.org/wiki/Discretization).

### IMM Model
This class can hold multiple `DynamicModel` objects and defines functions to calculate the probability of switching between the models. 

#### Usage
To instantiate a **Interacting Multiple Models (IMM) object**, you must provide three parameters:

1. **Hold Times Vector**: This vector should contain the expected time durations between switches for each model. The length of this vector must equal the number of models, denoted as `N`.

2. **Switching Probabilities Matrix**: This is an `N x N` matrix where each element at index `(i, j)` represents the probability of transitioning from model `i` to model `j`. It's crucial that each probability lies between 0 and 1, and that the sum of probabilities in each row equals 1. These probabilities define the likelihood of transitioning to a particular model given that a switch occurs, but they do not represent the overall probability of a switch happening.

3. **DynamicModel Objects**: A set of `DynamicModel` objects, one for each model in use.

It's important to note that the actual probability of switching from one model to another is determined through the `hold_times` vector. By treating the system as a **Continuous Time Markov Chain (CTMC)**, as detailed on [Wikipedia](https://en.wikipedia.org/wiki/Continuous-time_Markov_chain), the model calculates the switching probabilities based on the specified hold times and the switching probabilities matrix. 



### Predefined Model Implementations
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


