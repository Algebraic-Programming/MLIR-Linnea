#ifndef LINNEA_MUL_TEST
#define LINNEA_MUL_TEST

#include "Standalone/LinneaExpr.h"
#include "llvm/ADT/SmallVector.h"

using namespace std;
using namespace llvm;
using namespace mlir::linnea::expr;

namespace {
template <typename... Args>
SmallVector<typename std::common_type<Args...>::type>
varargToVector(Args... args) {
  SmallVector<typename std::common_type<Args...>::type> result;
  result.reserve(sizeof...(Args));
  for (auto arg :
       {static_cast<typename std::common_type<Args...>::type>(args)...}) {
    result.emplace_back(arg);
  }
  return result;
}

template <typename... Args>
Expr *adderImpl(bool fold, Args... args) {
  SmallVector<Expr *, 4> operands = varargToVector<Expr *>(args...);
  assert(operands.size() >= 2 && "two or more operands");
  // here we hard-code the semirings, as we have other tests for the semirings.
  return variadicAdd(operands, fold, NaryExpr::SemiringsKind::REAL_ARITH);
}

template <typename... Args>
Expr *adder(Args... args) {
  return adderImpl(/*fold*/ true, args...);
}

template <typename... Args>
Expr *adder(bool arg, Args... args) {
  return adderImpl(arg, args...);
}

template <typename... Args>
Expr *multiplierImpl(bool fold, Args... args) {
  SmallVector<Expr *> operands = varargToVector<Expr *>(args...);
  assert(operands.size() >= 2 && "two or more operands");
  // here we hard-code the semirings (see above).
  return variadicMul(operands, fold, NaryExpr::SemiringsKind::REAL_ARITH);
}

template <typename... Args>
Expr *multiplier(Args... args) {
  return multiplierImpl(/*fold*/ true, args...);
}

template <typename... Args>
Expr *multiplier(bool arg, Args... args) {
  return multiplierImpl(arg, args...);
}
} // end namespace

template <typename Arg, typename... Args>
Expr *mul(Arg arg, Args... args) {
  return multiplier(arg, args...);
}

template <typename Arg, typename... Args>
Expr *add(Arg arg, Args... args) {
  return adder(arg, args...);
}
#endif
