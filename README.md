Simple Kalman Filter for estimating objects in odom frame. Includes a general ekf implementation which can be configured for
use with generic process and sensor model.

Kalman Filter:

state vector:       x = [x, y, z, phi, theta, psi]
measurement vector: y = [x, y, z, phi, theta, psi]

process model:     x_dot = 0 + v
measurement model: y = I_6x6 * x


Important Files in ekf_python2:

dynamicmodels_py2.py: contains a parent class for dynamic models and as many different dynamic models as need to be defined 
as subclasses

measurementmodels_py2.py: contains a parent class for sensor models and as many different sensor models as need to be defined 
as subclasses

ekf_py2.py: contains a class implementing the ekf equations. In the case of linear model this will collapse to a standard Kalman Filter

gaussparams_py2.py: contains a class which implements multivariate gaussians

Pip3/pip requirement:
pip install dataclasses