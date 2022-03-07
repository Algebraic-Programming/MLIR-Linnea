#include "basicOp.h"
#include "gtest/gtest.h"

using namespace std;
using namespace mlir::linnea::expr;

TEST(Chain, MCP) {
  ScopedContext ctx;
  auto *a = new Matrix("A1", {30, 35});
  auto *b = new Matrix("A2", {35, 15});
  auto *c = new Matrix("A3", {15, 5});
  auto *d = new Matrix("A4", {5, 10});
  auto *e = new Matrix("A5", {10, 20});
  auto *f = new Matrix("A6", {20, 25});
  auto *g = static_cast<NaryExpr *>(mul(a, mul(b, mul(c, mul(d, mul(e, f))))));
  long result = g->getMCPFlops();
  EXPECT_EQ(result, 30250);
}

TEST(Chain, MCPVariadicMul) {
  ScopedContext ctx;
  auto *a = new Matrix("A1", {30, 35});
  auto *b = new Matrix("A2", {35, 15});
  auto *c = new Matrix("A3", {15, 5});
  auto *d = new Matrix("A4", {5, 10});
  auto *e = new Matrix("A5", {10, 20});
  auto *f = new Matrix("A6", {20, 25});
  auto *g = static_cast<NaryExpr *>(mul(a, b, c, d, e, f));

  long result = g->getMCPFlops();
  EXPECT_EQ(result, 30250);
}

// Expect cost to be n^2 * m * 2 -> 20 * 20 * 15 * 2
TEST(Chain, Cost) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  auto *b = new Matrix("B", {20, 15});
  auto *e = static_cast<NaryExpr *>(mul(a, b));
  long result = e->getMCPFlops();
  EXPECT_EQ(result, (20 * 20 * 15) << 1);
}

// Expect cost to be n^2 * m as A is lower triangular
// lower triangular are square, verify?
TEST(Chain, CostWithProperty) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  auto *b = new Matrix("B", {20, 15});
  a->setProperties({Expr::ExprProperty::LOWER_TRIANGULAR});
  auto *m = static_cast<NaryExpr *>(mul(a, b));
  long result = m->getMCPFlops();
  EXPECT_EQ(result, (20 * 20 * 15));
}

TEST(Chain, kernelCostWhenSPD) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  a->setProperties({Expr::ExprProperty::FULL_RANK});
  auto *b = new Matrix("B", {20, 15});
  long cost = 0;
  auto *e = static_cast<NaryExpr *>(mul(mul(trans(a), a), b));
  cost = e->getMCPFlops();
  EXPECT_EQ(cost, 22000);
}

TEST(Chain, CountFlopsIsSPD) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  a->setProperties({Expr::ExprProperty::FULL_RANK});
  auto *b = new Matrix("B", {20, 15});
  auto *e = static_cast<NaryExpr *>(mul(mul(trans(a), a), b));
  auto result = e->getMCPFlops();
  EXPECT_EQ(result, 22000);
}

TEST(Chain, CountFlopsIsSymmetric) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  auto *b = new Matrix("B", {20, 15});
  auto *e = static_cast<NaryExpr *>(mul(mul(trans(a), a), b));
  auto result = e->getMCPFlops();
  EXPECT_EQ(result, 22000);
  auto *f = static_cast<NaryExpr *>(mul(mul(a, trans(a)), b));
  result = f->getMCPFlops();
  EXPECT_EQ(result, 22000);
  auto *g = static_cast<NaryExpr *>(mul(a, trans(a), b));
  result = g->getMCPFlops();
  EXPECT_EQ(result, 22000);
}

// See table 1: https://arxiv.org/pdf/1804.04021.pdf
TEST(Chain, costBlas) {
  ScopedContext ctx;
  auto *x = new Matrix("A", {30, 20});
  auto *y = new Matrix("B", {30, 40});
  // GEMM
  auto *z = static_cast<NaryExpr *>(mul(x, y));
  auto flops = z->getMCPFlops();
  EXPECT_EQ(flops, 2 * 20 * 30 * 40);

  x = new Matrix("A", {30, 30});
  x->setProperties({Expr::ExprProperty::LOWER_TRIANGULAR});
  y = new Matrix("B", {30, 20});
  // TRMM
  z = static_cast<NaryExpr *>(mul(x, y));
  flops = z->getMCPFlops();
  EXPECT_EQ(flops, 30 * 30 * 20);

  x = new Matrix("A", {30, 30});
  x->setProperties({Expr::ExprProperty::SYMMETRIC});
  y = new Matrix("B", {30, 20});
  // SYMM
  z = static_cast<NaryExpr *>(mul(x, y));
  flops = z->getMCPFlops();
  EXPECT_EQ(flops, 30 * 30 * 20);
}

TEST(Chain, LinneaTest0) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {100, 90});
  auto *b = new Matrix("B", {90, 90});
  b->setProperties({Expr::ExprProperty::SPD});
  auto *c = new Matrix("C", {90, 80});
  auto *d = new Matrix("D", {80, 70});
  auto *x = static_cast<NaryExpr *>(mul(a, b, c, d));
  auto flops = x->getMCPFlops();
  EXPECT_EQ(flops, 3402000);
}
