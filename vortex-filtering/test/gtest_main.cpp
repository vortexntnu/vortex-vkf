#include <eigen3/Eigen/Core>
#include <gtest/gtest.h>

int main(int argc, char **argv)
{
  Eigen::initParallel();
  Eigen::setNbThreads(8);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}