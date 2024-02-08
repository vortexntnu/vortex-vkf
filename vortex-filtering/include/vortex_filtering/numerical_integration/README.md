# Integration Methods
A collection of integration methods. 
All methods can be used with all of the Kalman filter implementations.

## Use Cases
| Method        | Pros                      | Cons                                  | Use Case 
| ----          | ----                      | ----                                  | ----     
| Forward Euler | Fast                      | Most inaccurate                       | If speed is your only concern, use this
| RK4           | Accurate                  | 4 stages -> Slower                    | Should be the go-to method for most cases
| RK45          | As accurate as you want   | Adaptive step size -> Super slow      | If you need a set accuracy. Not meant for real-time applications
| Midpoint      | Stable                    | 2 stages -> Inaccurate                | Stable 2-stage method (compared to other 2-stage methods)
| Heun          |                           |                                       | Added just for fun :)
| Butcher       |                           | Slower than a direct implementation   | General Explicit Runge-Kutta method. Can be used to implement any explicit ERK method from a Butcher Table
