#include "basicOp.h"
#include "gtest/gtest.h"

using namespace std;
using namespace mlir::linnea::expr;

TEST(Dimensions, Matrix) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {30, 15});
  auto dims = a->getResultShape();
  EXPECT_EQ(dims.size() == 2, true);
  EXPECT_EQ(dims[0] == 30, true);
  EXPECT_EQ(dims[1] == 15, true);
}

TEST(Dimensions, Nary) {
  ScopedContext ctx;
  auto *a = new Matrix("A1", {30, 35});
  auto *b = new Matrix("A2", {35, 15});
  auto *c = new Matrix("A3", {15, 5});
  auto *d = new Matrix("A4", {5, 10});
  auto *e = new Matrix("A5", {10, 20});
  auto *f = new Matrix("A6", {20, 25});
  auto *g = mul(a, mul(b, mul(c, mul(d, mul(e, f)))));
  auto dims = g->getResultShape();
  EXPECT_EQ(dims.size() == 2, true);
  EXPECT_EQ(dims[0] == 30, true);
  EXPECT_EQ(dims[1] == 25, true);
}

TEST(Dimensions, Unary) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {30, 15});
  auto *at = trans(a);
  auto dims = at->getResultShape();
  EXPECT_EQ(dims.size() == 2, true);
  EXPECT_EQ(dims[0] == 15, true);
  EXPECT_EQ(dims[1] == 30, true);
}
