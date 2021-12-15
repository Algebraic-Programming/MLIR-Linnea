#include "mul.h"
#include "gtest/gtest.h"

using namespace std;
using namespace mlir::linnea::expr;

TEST(Dimensions, Operand) {
  ScopedContext ctx;
  auto *A = new Operand("A", {30, 15});
  auto dims = A->getResultDimensions();
  EXPECT_EQ(dims.size() == 2, true);
  EXPECT_EQ(dims[0] == 30, true);
  EXPECT_EQ(dims[1] == 15, true);
}

TEST(Dimensions, Nary) {
  ScopedContext ctx;
  auto *A = new Operand("A1", {30, 35});
  auto *B = new Operand("A2", {35, 15});
  auto *C = new Operand("A3", {15, 5});
  auto *D = new Operand("A4", {5, 10});
  auto *E = new Operand("A5", {10, 20});
  auto *F = new Operand("A6", {20, 25});
  auto *G = mul(A, mul(B, mul(C, mul(D, mul(E, F)))));
  auto dims = G->getResultDimensions();
  EXPECT_EQ(dims.size() == 2, true);
  EXPECT_EQ(dims[0] == 30, true);
  EXPECT_EQ(dims[1] == 25, true);
}

TEST(Dimensions, Unary) {
  ScopedContext ctx;
  auto *A = new Operand("A", {30, 15});
  auto *AT = trans(A);
  auto dims = AT->getResultDimensions();
  EXPECT_EQ(dims.size() == 2, true);
  EXPECT_EQ(dims[0] == 15, true);
  EXPECT_EQ(dims[1] == 30, true);
}
