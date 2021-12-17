#ifndef MLIR_LINNEA_EXPR_IMPL
#define MLIR_LINNEA_EXPR_IMPL

namespace mlir {
namespace linnea {
namespace expr {

template <NaryExpr::NaryExprKind K>
Expr *variadicExpr(std::vector<Expr *> children, bool isBinary) {
  if (isBinary) {
    assert(children.size() == 2 && "expect only two children");
    return new NaryExpr({children[0], children[1]},
                        NaryExpr::NaryExprKind::MUL);
  }
  // fold child expr inside.
  std::vector<Expr *> newChildren;
  int size = children.size();
  for (int i = size - 1; i >= 0; i--) {
    if (auto childExpr = llvm::dyn_cast_or_null<NaryExpr>(children.at(i))) {
      auto childrenOfChildExpr = childExpr->getChildren();
      newChildren.insert(newChildren.begin(), childrenOfChildExpr.begin(),
                         childrenOfChildExpr.end());
    } else
      newChildren.insert(newChildren.begin(), children.at(i));
  }
  return new NaryExpr(newChildren, K);
}

inline Expr *variadicMul(std::vector<Expr *> children, bool isBinary) {
  return variadicExpr<NaryExpr::NaryExprKind::MUL>(children, isBinary);
}

inline Expr *variadicAdd(std::vector<Expr *> children, bool isBinary) {
  return variadicExpr<NaryExpr::NaryExprKind::ADD>(children, isBinary);
}

template <UnaryExpr::UnaryExprKind K>
Expr *unaryExpr(Expr *child) {
  return new UnaryExpr(child, K);
}

inline Expr *inv(Expr *child) {
  return unaryExpr<UnaryExpr::UnaryExprKind::INVERSE>(child);
}

inline Expr *trans(Expr *child) {
  return unaryExpr<UnaryExpr::UnaryExpr::UnaryExprKind::TRANSPOSE>(child);
}

} // end namespace expr.
} // end namespace linnea.
} // end namespace mlir.

#endif
