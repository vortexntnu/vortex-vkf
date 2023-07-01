#include <gtest/gtest.h>
#include <models/LTI_model.hpp>
// #include <kalman_filters/EKF.hpp>

#include <random>
#include <vector>

class EKFTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        // Deterministic tests
        srand(0);
        for (size_t i{0}; i<100; i++)
        {
            
        }
    }
};