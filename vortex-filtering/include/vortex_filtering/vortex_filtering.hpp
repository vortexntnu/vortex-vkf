/**
 * @file vortex_filtering.hpp
 * @author Eirik Kolås
 * @brief File for all of the includes in the vortex-filtering library
 * @version 0.1
 * @date 2023-11-17
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

// Filters
#include <vortex_filtering/filters/ekf.hpp>
#include <vortex_filtering/filters/ukf.hpp>

// Models
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/dynamic_models.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_models.hpp>

// Numerical Integration
#include <vortex_filtering/numerical_integration/erk_methods.hpp>

// Utils
#include <vortex_filtering/utils/ellipse.hpp>
#include <vortex_filtering/utils/plotting.hpp>
#include <vortex_filtering/utils/printers.hpp>

// Probability
#include <vortex_filtering/probability/gaussian_mixture.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>
