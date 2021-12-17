#ifndef LINNEA_MUL_TEST
#define LINNEA_MUL_TEST

#include "Standalone/LinneaExpr.h"

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

template <typename Arg, typename... Args>
Expr *mul(Arg arg, Args... args) {
  auto operands = varargToVector<Expr *>(arg, args...);
  assert(operands.size() >= 2 && "two or more operands");
  return variadicMul(operands, /*fold*/ true);
}

template <typename Arg, typename... Args>
Expr *add(Arg arg, Args... args) {
  auto operands = varargToVector<Expr *>(arg, args...);
  assert(operands.size() >= 2 && "two or more operands");
  return variadicAdd(operands, /*fold*/ true);
}
#endif
