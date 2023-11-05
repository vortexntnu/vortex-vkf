# Models
This folder contains the models used in the Vortex-VKF project. The models are derived from either DynamicModelI or the SensorModelI interface. 
They define the dynamics of the system and the sensor models used in the project.

All classes and functions are under the namespace `vortex::models`.

## DynamicModelI
This interface defines the dynamics of the system. The methods `f_c`, `A_c` and `Q_c` define the continuous time dynamics of the system. The methods `f_d`, `A_d` and `Q_d` define the discrete time dynamics of the system. The methods `f_c` and `f_d` are the state transition functions. The methods `A_c` and `A_d` are the Jacobians of the state transition functions. The methods `Q_c` and `Q_d` are the process noise covariance matrices. 

### Usage
In order to define a new dynamic model, the user must create a new class that inherits from the DynamicModelI interface. The user must then implement the methods `f_c`, `A_c` and `Q_c` for the continuous time dynamics. The discrete time dynamics are found using exact discretization. If you want to use this, you must implement the methods `f_c`, `A_c` and `Q_c`. The methods `f_d`, `A_d` and `Q_d` are then automatically generated using [exact discretization](https://en.wikipedia.org/wiki/Discretization). The user can also define the discrete time dynamics themselves by overiting `f_d` etc. 

The interface is a template class with parameter `N_DIM_x`. This is the dimension of the state vector. The user must specify this when creating a new class, or derive a template class from the DynamicModelI interface. Both static and dynamic dimensions are supported but as of now only static dimensions are tested.

## SensorModelI
This interface defines the sensor models. The methods `h` `H` and `R` define the sensor model. The method `h` is the measurement function, `H` is the Jacobian of the measurement function and `R` is the measurement noise covariance matrix. The sensor model is assumed to be defined in discrete time.

### Usage
In order to define a new sensor model, the user must create a new class that inherits from the SensorModelI interface. The user must then implement the methods `h`, `H` and `R` as they are pure virtual.

The interface is a template class with parameter `N_DIM_x` and `N_DIM_z`. `N_DIM_x` is the dimension of the state vector and `N_DIM_z` is the dimension of the measurement vector. The user must specify these when creating a new class, or derive a template class from the SensorModelI interface. Both static and dynamic dimensions are supported (and tested).

## Movement Models
This file contains some movement models that can be used in the project. They are all derived from the DynamicModelI interface.
- **CVModel**: Constant Velocity Model. Has four states: x, y, vx and vy. The velocity is assumed to be constant.
- **CTModel**: Coordinated Turn Model. Has five states: x, y, vx, vy and omega. The speed is assumed to be constant and the turn rate is assumed to be constant.
- **CAModel**: Constant Acceleration Model. Has six states: x, y, vx, vy, ax and ay. The acceleration is assumed to be constant.

## IMM Model
This class is can hold multiple DynamicModelI objects and defines functions to calculate the probability of switching between the models. 

### Usage
To create an instance you need to provide a vector DynamicModelI objects and a matrix defining the switching probabilities. The switching probabilities are defined as the probability of switching from model i to model j. The switching probabilities must be between 0 and 1 and the rows must sum to 1.0. The number of rows and columns must be the same as the number of models. Note that this defines the probability of switching to a specific model *when* a switch occurs. Not the probability of switching itself. The probability of switching is derived from the `hold_times` parameter. The `hold_times` parameter is a vector of the expected time between switches for each model. The switching probabilities are then calculated by modelling the system as a [Continuous Time Markov Chain](https://en.wikipedia.org/wiki/Continuous-time_Markov_chain). The switching probabilities are then used to calculate the probability of switching or staying in a model.
