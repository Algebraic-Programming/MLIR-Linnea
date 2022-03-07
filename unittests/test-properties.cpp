#include "basicOp.h"
#include "gtest/gtest.h"

using namespace std;
using namespace mlir::linnea::expr;

// must fails: a triangular matrix is a special kind of square matrix.
TEST(Property, LowerTriangular) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 10});
  a->setProperties({Expr::ExprProperty::LOWER_TRIANGULAR});
  EXPECT_EQ(a->isLowerTriangular(), false);
}

// must fails: a symmetric matrix is a square matrix that is equal to its
// transpose.
TEST(Property, Symmetric) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 10});
  a->setProperties({Expr::ExprProperty::SYMMETRIC});
  EXPECT_EQ(a->isSymmetric(), false);
}

TEST(Property, TriangularInverse) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  a->setProperties(
      {Expr::ExprProperty::LOWER_TRIANGULAR, Expr::ExprProperty::FULL_RANK});
  auto *expr = inv(a);
  EXPECT_EQ(expr->isLowerTriangular(), true);
}

TEST(Property, InferPropertyWhenBuildingObj) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  EXPECT_EQ(a->isSquare(), true);
  auto *b = new Matrix("B", {30, 20});
  EXPECT_EQ(b->isSquare(), false);
}

// Any matrix congruent to a symmetric matrix is again symmetric
TEST(Property, CongruentMul) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {30, 20});
  auto *b = new Matrix("B", {20, 20});
  b->setProperties({Expr::ExprProperty::SYMMETRIC});
  auto *expr = mul(a, b, trans(a));
  EXPECT_EQ(expr->isSymmetric(), true);
}

// If A is Symmetric so is A^n
TEST(Property, SymmetricPower) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  a->setProperties({Expr::ExprProperty::SYMMETRIC});
  auto *expr = mul(a, a, a);
  EXPECT_EQ(expr->isSymmetric(), true);
}

// If A is Lower (Upper) triangular then also A^n has the same property
TEST(Property, TriangularPowerLower) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  a->setProperties({Expr::ExprProperty::LOWER_TRIANGULAR});
  auto *expr = mul(a, a, a);
  EXPECT_EQ(expr->isLowerTriangular(), true);
  EXPECT_EQ(trans(expr)->isUpperTriangular(), true);
}

TEST(Property, TriangularPowerUpper) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  a->setProperties({Expr::ExprProperty::UPPER_TRIANGULAR});
  auto *expr = mul(a, a, a);
  EXPECT_EQ(expr->isUpperTriangular(), true);
  EXPECT_EQ(trans(expr)->isLowerTriangular(), true);
}

// Expect lowerTriangular as all operands of the mul are LT.
TEST(Property, AllLowerTriangular) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  auto *b = new Matrix("B", {20, 20});
  a->setProperties({Expr::ExprProperty::LOWER_TRIANGULAR});
  b->setProperties({Expr::ExprProperty::LOWER_TRIANGULAR});
  auto *expr = mul(a, b, a);
  EXPECT_EQ(expr->isLowerTriangular(), true);
  EXPECT_EQ(trans(expr)->isUpperTriangular(), true);
}

// The product of two upper (lower) triangular matrices is upper (lower)
// triangular matrix.
TEST(Property, PropagationRulesUpperTimesUpper) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  a->setProperties({Expr::ExprProperty::UPPER_TRIANGULAR});
  auto *b = new Matrix("B", {20, 20});
  b->setProperties({Expr::ExprProperty::UPPER_TRIANGULAR});
  auto *m = mul(a, b);
  EXPECT_EQ(m->isUpperTriangular(), true);
}

TEST(Property, PropagationRulesLowerTimesLower) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  a->setProperties({Expr::ExprProperty::LOWER_TRIANGULAR});
  auto *b = new Matrix("B", {20, 20});
  b->setProperties({Expr::ExprProperty::LOWER_TRIANGULAR});
  auto *aTimesB = mul(a, b);
  EXPECT_EQ(aTimesB->isLowerTriangular(), true);
  auto *aTimesBTransTrans = mul(a, trans(trans(b)));
  EXPECT_EQ(aTimesBTransTrans->isLowerTriangular(), true);
  auto *aTransTransTimesB = mul(trans(trans(a)), b);
  EXPECT_EQ(aTransTransTimesB->isLowerTriangular(), true);
}

// If you transpose an upper (lower) triangular matrix, you get a lower (upper)
// triangular matrix.
TEST(Property, PropagationRulesTransposeUpper) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  a->setProperties({Expr::ExprProperty::UPPER_TRIANGULAR});
  auto *t = trans(a);
  EXPECT_EQ(t->isLowerTriangular(), true);
}

TEST(Property, PropagationRulesTransposeLower) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  a->setProperties({Expr::ExprProperty::LOWER_TRIANGULAR});
  auto *t = trans(a);
  EXPECT_EQ(t->isUpperTriangular(), true);
}

TEST(Property, PropagationRulesTransposeMultipleTimes) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  a->setProperties({Expr::ExprProperty::UPPER_TRIANGULAR});
  auto *t = trans(trans(a));
  EXPECT_EQ(t->isUpperTriangular(), true);
  t = trans(trans(trans(a)));
  EXPECT_EQ(t->isLowerTriangular(), true);
}

TEST(Property, PropagationRulesIsFullRank) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  a->setProperties({Expr::ExprProperty::FULL_RANK});
  auto *t = trans(a);
  EXPECT_EQ(t->isFullRank(), true);
  auto *i = inv(a);
  EXPECT_EQ(i->isFullRank(), true);
  auto *it = inv(trans(a));
  EXPECT_EQ(it->isFullRank(), true);
}

TEST(Property, PropagationRulesIsSPD) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  a->setProperties({Expr::ExprProperty::FULL_RANK});
  auto *spd = mul(trans(a), a);
  EXPECT_EQ(spd->isSPD(), true);
}

TEST(Property, SymmetricInverse) {
  ScopedContext ctx;
  auto *a = new Matrix("A", {20, 20});
  a->setProperties(
      {Expr::ExprProperty::SYMMETRIC, Expr::ExprProperty::FULL_RANK});
  auto *expr = inv(a);
  EXPECT_EQ(expr->isSymmetric(), true);
}
