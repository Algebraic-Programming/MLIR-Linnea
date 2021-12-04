#include "Standalone/LinneaExpr.h"
#include "gtest/gtest.h"

using namespace std;
using namespace mlir::linnea::expr;

namespace {
template <typename... Args>
vector<typename std::common_type<Args...>::type> varargToVector(Args... args) {
  vector<typename std::common_type<Args...>::type> result;
  result.reserve(sizeof...(Args));
  for (auto arg :
       {static_cast<typename std::common_type<Args...>::type>(args)...}) {
    result.emplace_back(arg);
  }
  return result;
}
} // end namespace

template <typename Arg, typename... Args> Expr *mul(Arg arg, Args... args) {
  auto operands = varargToVector<Expr *>(arg, args...);
  assert(operands.size() >= 2 && "one or more operands");
  return variadicMul(operands);
}

TEST(Chain, MCP) {
  ScopedContext ctx;
  auto *A = new Operand("A1", {30, 35});
  auto *B = new Operand("A2", {35, 15});
  auto *C = new Operand("A3", {15, 5});
  auto *D = new Operand("A4", {5, 10});
  auto *E = new Operand("A5", {10, 20});
  auto *F = new Operand("A6", {20, 25});
  auto G = mul(A, mul(B, mul(C, mul(D, mul(E, F)))));
  long result = G->getMCPFlops();
  EXPECT_EQ(result, 30250);
}

TEST(Chain, MCPVariadicMul) {
  ScopedContext ctx;
  auto *A = new Operand("A1", {30, 35});
  auto *B = new Operand("A2", {35, 15});
  auto *C = new Operand("A3", {15, 5});
  auto *D = new Operand("A4", {5, 10});
  auto *E = new Operand("A5", {10, 20});
  auto *F = new Operand("A6", {20, 25});
  auto G = mul(A, B, C, D, E, F);
  long result = G->getMCPFlops();
  EXPECT_EQ(result, 30250);
}

// Expect cost to be n^2 * m * 2 -> 20 * 20 * 15 * 2
TEST(Chain, Cost) {
  ScopedContext ctx;
  auto *A = new Operand("A", {20, 20});
  auto *B = new Operand("B", {20, 15});
  auto *E = mul(A, B);
  long result = E->getMCPFlops();
  EXPECT_EQ(result, (20 * 20 * 15) << 1);
}

// Expect cost to be n^2 * m as A is lower triangular
// lower triangular are square, verify?
TEST(Chain, CostWithProperty) {
  ScopedContext ctx;
  auto *A = new Operand("A", {20, 20});
  auto *B = new Operand("B", {20, 15});
  A->setProperties({Expr::ExprProperty::LOWER_TRIANGULAR});
  auto *M = mul(A, B);
  long result = M->getMCPFlops();
  EXPECT_EQ(result, (20 * 20 * 15));
}

TEST(Chain, kernelCostWhenSPD) {
  ScopedContext ctx;
  auto *A = new Operand("A", {20, 20});
  auto *B = new Operand("B", {20, 15});
  A->setProperties({Expr::ExprProperty::FULL_RANK});
  long cost = 0;
  auto E = mul(mul(trans(A), A), B);
  cost = E->getMCPFlops();
  EXPECT_EQ(cost, 22000);
}

TEST(Chain, CountFlopsIsSPD) {
  ScopedContext ctx;
  auto *A = new Operand("A", {20, 20});
  auto *B = new Operand("B", {20, 15});
  A->setProperties({Expr::ExprProperty::FULL_RANK});
  auto E = mul(mul(trans(A), A), B);
  auto result = E->getMCPFlops();
  EXPECT_EQ(result, 22000);
}

TEST(Chain, CountFlopsIsSymmetric) {
  ScopedContext ctx;
  auto *A = new Operand("A", {20, 20});
  auto *B = new Operand("B", {20, 15});
  auto E = mul(mul(trans(A), A), B);
  auto result = E->getMCPFlops();
  EXPECT_EQ(result, 22000);
  auto F = mul(mul(A, trans(A)), B);
  result = F->getMCPFlops();
  EXPECT_EQ(result, 22000);
  auto G = mul(A, trans(A), B);
  result = G->getMCPFlops();
  EXPECT_EQ(result, 22000);
}
