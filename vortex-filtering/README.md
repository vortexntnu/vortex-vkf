# vortex-filtering
## Models
Contains the models used in the filters. The models are implemented as classes that inherit from the `DynamicModelBase` class or `SensorModelBase` class. The models are implemented in the `models` namespace. [More info](include/vortex_filtering/models/README.md)

## Filters
Contains the filters. The filters are implemented as classes that inherit from the `KalmanFilterBase` class. The filters are implemented in the `filters` namespace. [More info](include/vortex_filtering/filters/README.md)

## Class Diagram

```mermaid
classDiagram

    DynamicModel <|-- DynamicModelLTV
    SensorModel <|-- SensorModelLTV
    DynamicModelLTV <|-- ConstantVelocity
    DynamicModelLTV <|-- ConstantAcceleration
    DynamicModelLTV <|-- CoordinatedTurn

    EKF -- DynamicModelLTV
    EKF -- SensorModelLTV

    UKF -- DynamicModel
    UKF -- SensorModel



    class EKF{
        +predict()
        +update()
        +step()
    }

    class UKF{
        +predict()
        +update()
        +step()
        -get_sigma_points()
        -propagate_sigma_points_f()
        -propagate_sigma_points_h()
        -estimate_gaussian()
    }

    class DynamicModel{
        +virtual f_d() Vec_x
        +virtual Q_d() Mat_vv
        +sample_f_d() Vec_x
    }

    class SensorModel{
        +virtual h() Vec_z
        +virtual R() Mat_ww
        +sample_h() Vec_z
    }

    class DynamicModelLTV {
        +overide f_d() Vec_x
        +virtual A_d() Mat_xx
        +virtual Q_d() Mat_vv
        +vurtual G_d() Mat_xv
        +pred_from_est() Gauss_x
        +pred_from_state() Gauss_x
    }

    class SensorModelLTV {
        +override h(x) Vec_z
        +virtual R(x) Mat_ww
        +virtual C(x) Mat_zx
        +virtual H(x) Mat_zw
        +pred_from_est(x_est) Gauss_z
        +pred_from_state(x) Gauss_z
    }

    class ConstantVelocity
    class CoordinatedTurn
    class ConstantAcceleration

```