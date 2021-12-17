#ifndef MLIR_LINNEA_EXPR_IMPL
#define MLIR_LINNEA_EXPR_IMPL

namespace mlir {
namespace linnea {
namespace expr {

template <NaryExpr::NaryExprKind K>
Expr *variadicExpr(std::vector<Expr *> children, bool fold) {
  if (fold) {
    // fold child expr inside.
    std::vector<Expr *> newChildren;
    int size = children.size();
    for (int i = size - 1; i >= 0; i--) {
      if (auto childExpr = llvm::dyn_cast_or_null<NaryExpr>(children.at(i))) {
        if (childExpr->getKind() != K) {
          newChildren.insert(newChildren.begin(), children.at(i));
          continue;
        }
        auto childrenOfChildExpr = childExpr->getChildren();
        newChildren.insert(newChildren.begin(), childrenOfChildExpr.begin(),
                           childrenOfChildExpr.end());
      } else
        newChildren.insert(newChildren.begin(), children.at(i));
    }
    return new NaryExpr(newChildren, K);
  }
  assert(children.size() >= 2 && "two or more children");
  Expr *result = new NaryExpr({children[0], children[1]}, K);
  for (size_t i = 2; i < children.size(); i++) {
    result = new NaryExpr({result, children[i]}, K);
  }
  return result;
}

inline Expr *variadicMul(std::vector<Expr *> children, bool fold) {
  return variadicExpr<NaryExpr::NaryExprKind::MUL>(children, fold);
}

inline Expr *variadicAdd(std::vector<Expr *> children, bool fold) {
  return variadicExpr<NaryExpr::NaryExprKind::ADD>(children, fold);
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
