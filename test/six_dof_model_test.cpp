#include <gtest/gtest.h>

#include <models/SIX_DOF_model.hpp>
using namespace Models;

class SIX_DOF_modelTest : public ::testing::Test {
protected:
    static constexpr int N_3D_VECS{2}, N_QUAT{1}, N_INPUTS{1}, N_MEAS{3};
    SIX_DOF_modelTest() : model{}
    {
        model = SIX_DOF_model<N_3D_VECS,N_QUAT,N_INPUTS,N_MEAS>{};
    }

    SIX_DOF_model<N_3D_VECS,N_QUAT,N_INPUTS,N_MEAS> model;
};