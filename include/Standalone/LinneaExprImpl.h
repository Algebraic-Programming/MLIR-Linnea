//===- LinneaExprImpl.h ----------------------------------------*- C++ -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LINNEA_EXPR_IMPL
#define MLIR_LINNEA_EXPR_IMPL

namespace mlir {
namespace linnea {
namespace expr {

template <NaryExpr::NaryExprKind K>
Expr *variadicExpr(ArrayRef<Expr *> children, bool fold,
                   NaryExpr::SemiringsKind S) {
  if (fold) {
    // fold child expr inside.
    std::vector<Expr *> newChildren;
    int size = children.size();
    for (int i = size - 1; i >= 0; i--) {
      if (auto childExpr = llvm::dyn_cast_or_null<NaryExpr>(children[i])) {
        if (childExpr->getKind() != K) {
          newChildren.insert(newChildren.begin(), children[i]);
          continue;
        }
        auto childrenOfChildExpr = childExpr->getChildren();
        newChildren.insert(newChildren.begin(), childrenOfChildExpr.begin(),
                           childrenOfChildExpr.end());
      } else
        newChildren.insert(newChildren.begin(), children[i]);
    }
    return new NaryExpr(newChildren, K, S);
  }
  assert(children.size() >= 2 && "two or more children");
  Expr *result = new NaryExpr({children[0], children[1]}, K, S);
  for (size_t i = 2; i < children.size(); i++) {
    result = new NaryExpr({result, children[i]}, K, S);
  }
  return result;
}

inline Expr *variadicMul(ArrayRef<Expr *> children, bool fold,
                         NaryExpr::SemiringsKind S) {
  return variadicExpr<NaryExpr::NaryExprKind::MUL>(children, fold, S);
}

inline Expr *variadicAdd(ArrayRef<Expr *> children, bool fold,
                         NaryExpr::SemiringsKind S) {
  return variadicExpr<NaryExpr::NaryExprKind::ADD>(children, fold, S);
}

template <UnaryExpr::UnaryExprKind K>
Expr *unaryExpr(Expr *child) {
  return new UnaryExpr(child, K);
}

inline Expr *inv(Expr *child) {
  return unaryExpr<UnaryExpr::UnaryExprKind::INVERSE>(child);
}

inline Expr *trans(Expr *child) {
  return unaryExpr<UnaryExpr::UnaryExprKind::TRANSPOSE>(child);
}

} // end namespace expr.
} // end namespace linnea.
} // end namespace mlir.

#endif
