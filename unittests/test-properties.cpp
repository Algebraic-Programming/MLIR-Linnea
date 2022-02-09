#include "basicOp.h"
#include "gtest/gtest.h"

using namespace std;
using namespace mlir::linnea::expr;

// must fails: a triangular matrix is a special kind of square matrix.
TEST(Property, LowerTriangular) {
  ScopedContext ctx;
  auto *A = new Matrix("A", {20, 10});
  A->setProperties({Expr::ExprProperty::LOWER_TRIANGULAR});
  EXPECT_EQ(A->isLowerTriangular(), false);
}

// must fails: a symmetric matrix is a square matrix that is equal to its
// transpose.
TEST(Property, Symmetric) {
  ScopedContext ctx;
  auto *A = new Matrix("A", {20, 10});
  A->setProperties({Expr::ExprProperty::SYMMETRIC});
  EXPECT_EQ(A->isSymmetric(), false);
}

TEST(Property, TriangularInverse) {
  ScopedContext ctx;
  auto *A = new Matrix("A", {20, 20});
  A->setProperties(
      {Expr::ExprProperty::LOWER_TRIANGULAR, Expr::ExprProperty::FULL_RANK});
  auto expr = inv(A);
  EXPECT_EQ(expr->isLowerTriangular(), true);
}

TEST(Property, InferPropertyWhenBuildingObj) {
  ScopedContext ctx;
  auto *A = new Matrix("A", {20, 20});
  EXPECT_EQ(A->isSquare(), true);
  auto *B = new Matrix("B", {30, 20});
  EXPECT_EQ(B->isSquare(), false);
}

// Any matrix congruent to a symmetric matrix is again symmetric
TEST(Property, CongruentMul) {
  ScopedContext ctx;
  auto *A = new Matrix("A", {30, 20});
  auto *B = new Matrix("B", {20, 20});
  B->setProperties({Expr::ExprProperty::SYMMETRIC});
  auto expr = mul(A, B, trans(A));
  EXPECT_EQ(expr->isSymmetric(), true);
}

// If A is Symmetric so is A^n
TEST(Property, SymmetricPower) {
  ScopedContext ctx;
  auto *A = new Matrix("A", {20, 20});
  A->setProperties({Expr::ExprProperty::SYMMETRIC});
  auto expr = mul(A, A, A);
  EXPECT_EQ(expr->isSymmetric(), true);
}

// If A is Lower (Upper) triangular then also A^n has the same property
TEST(Property, TriangularPowerLower) {
  ScopedContext ctx;
  auto *A = new Matrix("A", {20, 20});
  A->setProperties({Expr::ExprProperty::LOWER_TRIANGULAR});
  auto *expr = mul(A, A, A);
  EXPECT_EQ(expr->isLowerTriangular(), true);
  EXPECT_EQ(trans(expr)->isUpperTriangular(), true);
}

TEST(Property, TriangularPowerUpper) {
  ScopedContext ctx;
  auto *A = new Matrix("A", {20, 20});
  A->setProperties({Expr::ExprProperty::UPPER_TRIANGULAR});
  auto *expr = mul(A, A, A);
  EXPECT_EQ(expr->isUpperTriangular(), true);
  EXPECT_EQ(trans(expr)->isLowerTriangular(), true);
}

// Expect lowerTriangular as all operands of the mul are LT.
TEST(Property, AllLowerTriangular) {
  ScopedContext ctx;
  auto *A = new Matrix("A", {20, 20});
  auto *B = new Matrix("B", {20, 20});
  A->setProperties({Expr::ExprProperty::LOWER_TRIANGULAR});
  B->setProperties({Expr::ExprProperty::LOWER_TRIANGULAR});
  auto expr = mul(A, B, A);
  EXPECT_EQ(expr->isLowerTriangular(), true);
  EXPECT_EQ(trans(expr)->isUpperTriangular(), true);
}

// The product of two upper (lower) triangular matrices is upper (lower)
// triangular matrix.
TEST(Property, PropagationRulesUpperTimesUpper) {
  ScopedContext ctx;
  auto *A = new Matrix("A", {20, 20});
  A->setProperties({Expr::ExprProperty::UPPER_TRIANGULAR});
  auto *B = new Matrix("B", {20, 20});
  B->setProperties({Expr::ExprProperty::UPPER_TRIANGULAR});
  auto *M = mul(A, B);
  EXPECT_EQ(M->isUpperTriangular(), true);
}

TEST(Property, PropagationRulesLowerTimesLower) {
  ScopedContext ctx;
  auto *A = new Matrix("A", {20, 20});
  A->setProperties({Expr::ExprProperty::LOWER_TRIANGULAR});
  auto *B = new Matrix("B", {20, 20});
  B->setProperties({Expr::ExprProperty::LOWER_TRIANGULAR});
  auto aTimesB = mul(A, B);
  EXPECT_EQ(aTimesB->isLowerTriangular(), true);
  auto aTimesBTransTrans = mul(A, trans(trans(B)));
  EXPECT_EQ(aTimesBTransTrans->isLowerTriangular(), true);
  auto aTransTransTimesB = mul(trans(trans(A)), B);
  EXPECT_EQ(aTransTransTimesB->isLowerTriangular(), true);
}

// If you transpose an upper (lower) triangular matrix, you get a lower (upper)
// triangular matrix.
TEST(Property, PropagationRulesTransposeUpper) {
  ScopedContext ctx;
  auto *A = new Matrix("A", {20, 20});
  A->setProperties({Expr::ExprProperty::UPPER_TRIANGULAR});
  auto T = trans(A);
  EXPECT_EQ(T->isLowerTriangular(), true);
}

TEST(Property, PropagationRulesTransposeLower) {
  ScopedContext ctx;
  auto *A = new Matrix("A", {20, 20});
  A->setProperties({Expr::ExprProperty::LOWER_TRIANGULAR});
  auto T = trans(A);
  EXPECT_EQ(T->isUpperTriangular(), true);
}

TEST(Property, PropagationRulesTransposeMultipleTimes) {
  ScopedContext ctx;
  auto *A = new Matrix("A", {20, 20});
  A->setProperties({Expr::ExprProperty::UPPER_TRIANGULAR});
  auto T = trans(trans(A));
  EXPECT_EQ(T->isUpperTriangular(), true);
  T = trans(trans(trans(A)));
  EXPECT_EQ(T->isLowerTriangular(), true);
}

TEST(Property, PropagationRulesIsFullRank) {
  ScopedContext ctx;
  auto *A = new Matrix("A", {20, 20});
  A->setProperties({Expr::ExprProperty::FULL_RANK});
  auto T = trans(A);
  EXPECT_EQ(T->isFullRank(), true);
  auto I = inv(A);
  EXPECT_EQ(I->isFullRank(), true);
  auto IT = inv(trans(A));
  EXPECT_EQ(IT->isFullRank(), true);
}

TEST(Property, PropagationRulesIsSPD) {
  ScopedContext ctx;
  auto *A = new Matrix("A", {20, 20});
  A->setProperties({Expr::ExprProperty::FULL_RANK});
  auto SPD = mul(trans(A), A);
  EXPECT_EQ(SPD->isSPD(), true);
}

TEST(Property, SymmetricInverse) {
  ScopedContext ctx;
  auto *A = new Matrix("A", {20, 20});
  A->setProperties(
      {Expr::ExprProperty::SYMMETRIC, Expr::ExprProperty::FULL_RANK});
  auto expr = inv(A);
  EXPECT_EQ(expr->isSymmetric(), true);
}
