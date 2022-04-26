//===- LinneaProperties.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaExpr.h"
#include "llvm/Support/Casting.h"
#include <algorithm>

using namespace mlir::linnea::expr;
using namespace llvm;
using namespace std;

//===----------------------------------------------------------------------===//
// Matrix
//===----------------------------------------------------------------------===//

template <Expr::ExprProperty P>
bool isX(const Operand *operand) {
  auto inferredProperties = operand->getProperties();
  return std::any_of(inferredProperties.begin(), inferredProperties.end(),
                     [](Expr::ExprProperty p) { return p == P; });
}

bool Matrix::isUpperTriangular() const {
  return isX<ExprProperty::UPPER_TRIANGULAR>(this);
}

bool Matrix::isLowerTriangular() const {
  return isX<ExprProperty::LOWER_TRIANGULAR>(this);
}

bool Matrix::isSquare() const { return isX<ExprProperty::SQUARE>(this); }

bool Matrix::isSymmetric() const { return isX<ExprProperty::SYMMETRIC>(this); }

bool Matrix::isFullRank() const { return isX<ExprProperty::FULL_RANK>(this); }

bool Matrix::isSPD() const { return isX<ExprProperty::SPD>(this); }

bool Matrix::isFactored() const { return isX<ExprProperty::FACTORED>(this); }

bool Matrix::isGeneral() const { return isX<ExprProperty::GENERAL>(this); }

//===----------------------------------------------------------------------===//
// Identity
//===----------------------------------------------------------------------===//

bool Identity::isUpperTriangular() const { return false; }

bool Identity::isLowerTriangular() const { return false; }

bool Identity::isSquare() const { return isX<ExprProperty::SQUARE>(this); }

bool Identity::isSymmetric() const { return false; }

bool Identity::isFullRank() const { return false; }

bool Identity::isSPD() const { return false; }

bool Identity::isFactored() const { return true; }

bool Identity::isGeneral() const { return true; }

//===----------------------------------------------------------------------===//
// UnaryExpr
//===----------------------------------------------------------------------===//

bool UnaryExpr::isGeneral() const { return child->isGeneral(); }

bool UnaryExpr::isFactored() const { return child->isFactored(); }

bool UnaryExpr::isUpperTriangular() const {
  auto kind = this->getKind();
  switch (kind) {
  case UnaryExprKind::TRANSPOSE:
    return child->isLowerTriangular();
  default:
    assert(0 && "UNK");
  }
  return false;
}

bool UnaryExpr::isLowerTriangular() const {
  auto kind = this->getKind();
  switch (kind) {
  case UnaryExprKind::TRANSPOSE:
    return child->isUpperTriangular();
  case UnaryExprKind::INVERSE:
    // TODO: do we need also the full rank?
    return child->isLowerTriangular() && child->isFullRank();
  }
  return false;
}

bool UnaryExpr::isSquare() const {
  auto kind = this->getKind();
  switch (kind) {
  case UnaryExprKind::TRANSPOSE:
  case UnaryExprKind::INVERSE:
    return child->isSquare();
  }
  return false;
}

bool UnaryExpr::isSymmetric() const {
  auto kind = this->getKind();
  switch (kind) {
  case UnaryExprKind::TRANSPOSE:
  case UnaryExprKind::INVERSE:
    return child->isSymmetric() || child->isSPD();
  }
  return false;
}

bool UnaryExpr::isFullRank() const {
  auto kind = this->getKind();
  switch (kind) {
  case UnaryExprKind::TRANSPOSE:
  case UnaryExprKind::INVERSE:
    return child->isFullRank();
  }
  return false;
}

bool UnaryExpr::isSPD() const {
  assert(0 && "no impl");
  return false;
}

//===----------------------------------------------------------------------===//
// NaryExpr
//===----------------------------------------------------------------------===//

// XXX: this is not true either. For exmaple
// mul(lower_tri * upper_tri) is general.
bool NaryExpr::isGeneral() const {
  auto kind = this->getKind();
  switch (kind) {
  // all the children must be general.
  case NaryExpr::NaryExprKind::MUL:
  case NaryExpr::NaryExprKind::ADD: {
    for (auto *child : this->getChildren()) {
      if (!child->isGeneral())
        return false;
    }
    return true;
  }
  }
  llvm_unreachable("Only MUL or ADD are supported");
  return false;
}

bool NaryExpr::isFactored() const { return false; }

bool NaryExpr::isUpperTriangular() const {
  auto kind = this->getKind();
  switch (kind) {
  // all the children must be upper triangular.
  case NaryExpr::NaryExprKind::MUL:
  case NaryExpr::NaryExprKind::ADD: {
    for (auto *child : this->getChildren()) {
      if (!child->isUpperTriangular())
        return false;
    }
    return true;
  }
  }
  llvm_unreachable("Only MUL or ADD are supported");
}

bool NaryExpr::isLowerTriangular() const {
  auto kind = this->getKind();
  switch (kind) {
  // all the children must be lower triangular.
  case NaryExpr::NaryExprKind::MUL:
  case NaryExpr::NaryExprKind::ADD: {
    for (auto *child : this->getChildren()) {
      if (!child->isLowerTriangular())
        return false;
    }
    return true;
  }
  }
  llvm_unreachable("Only MUL or ADD are supported");
}

bool NaryExpr::isSquare() const {
  // XXX: Fix this.
  // [5x3] * [3x5] = [5x5]
  // This is not true. Not all the children
  // need to be square.
  for (auto *child : this->getChildren())
    if (!child->isSquare())
      return false;
  return true;
}

/// Check if the product is full rank.
/// All children need to be full rank or square.
/// Multiplication by a full-rank square matrix preserves rank.
static bool isFullRankProduct(const NaryExpr *op) {
  assert(op->getKind() == NaryExpr::NaryExprKind::MUL && "must be mul");
  for (auto *child : op->getChildren()) {
    if (!child->isFullRank() && !child->isSquare())
      return false;
  }
  return true;
}

bool NaryExpr::isFullRank() const {
  auto kind = this->getKind();
  switch (kind) {
  case NaryExpr::NaryExprKind::MUL:
    return isFullRankProduct(this);
  default:
    assert(0 && "UNK");
  }
  llvm_unreachable("Only MUL supported");
}

static inline NaryExpr::SemiringsKind
getSemiringsKindFromExpr(const NaryExpr *op) {
  return op->getSemiringsKind();
}

// is the current product SPD?
static bool isSymmetricProduct(const NaryExpr *op, bool checkSPD = false) {
  assert(op->getKind() == NaryExpr::NaryExprKind::MUL && "must be mul");
  auto children = op->getChildren();
  if (children.size() == 1)
    return op->getChildren()[0]->isSPD();
  else if (children.size() % 2 == 0) {
    size_t middle = children.size() / 2;
    SmallVector<Expr *, 4> leftChildren;
    SmallVector<Expr *, 4> rightChildren;
    for (size_t i = 0; i < children.size(); i++) {
      if (i < middle)
        leftChildren.push_back(children[i]);
      else
        rightChildren.push_back(children[i]);
    }
    auto *leftExpr =
        variadicMul(leftChildren, /*fold*/ true, getSemiringsKindFromExpr(op));
    auto *rightExpr =
        variadicMul(rightChildren, /*fold*/ true, getSemiringsKindFromExpr(op));
#if DEBUG
    cout << __func__ << "\n";
    walk(leftExpr);
    cout << "\n";
    walk(rightExpr);
    cout << "\n";
    cout << "is left expr full rank: " << leftExpr->isFullRank() << "\n";
    cout << "is left expr square: " << leftExpr->isSquare() << "\n";
    cout << "is left expr transpose of right expr: "
         << leftExpr->isTransposeOf(rightExpr) << "\n";
#endif
    // if we check SPD, also check full rank and square.
    if (checkSPD)
      return leftExpr->isFullRank() && leftExpr->isSquare() &&
             leftExpr->isTransposeOf(rightExpr);
    else
      return leftExpr->isTransposeOf(rightExpr);
  } else {
    size_t middle = children.size() / 2;
    SmallVector<Expr *, 4> leftChildren;
    SmallVector<Expr *, 4> rightChildren;
    Expr *middleExpr = nullptr;
    for (size_t i = 0; i < children.size(); i++) {
      if (i < middle)
        leftChildren.push_back(children[i]);
      else if (i == middle)
        middleExpr = children[i];
      else
        rightChildren.push_back(children[i]);
    }
    auto *leftExpr =
        variadicMul(leftChildren, /*fold*/ true, getSemiringsKindFromExpr(op));
    auto *rightExpr =
        variadicMul(rightChildren, /*fold*/ true, getSemiringsKindFromExpr(op));
#if DEBUG
    cout << __func__ << "\n";
    walk(leftExpr);
    cout << "\n";
    walk(rightExpr);
    cout << "\n";
    cout << "is left expr full rank: " << leftExpr->isFullRank() << "\n";
    cout << "is left expr square: " << leftExpr->isSquare() << "\n";
    cout << "is left expr transpose of right expr: "
         << leftExpr->isTransposeOf(rightExpr) << "\n";
#endif
    if (checkSPD)
      return leftExpr->isFullRank() && leftExpr->isSquare() &&
             middleExpr->isSPD() && leftExpr->isTransposeOf(rightExpr);
    else
      return middleExpr->isSymmetric() && leftExpr->isTransposeOf(rightExpr);
  }
  return false;
}

bool NaryExpr::isSPD() const {
  auto kind = this->getKind();
  switch (kind) {
  case NaryExpr::NaryExprKind::MUL:
    return isSymmetricProduct(this, true);
  // TODO: implement me.
  case NaryExpr::NaryExprKind::ADD:
    return false;
  }
  llvm_unreachable("Only MUL supported");
}

bool NaryExpr::isSymmetric() const {
  auto kind = this->getKind();
  switch (kind) {
  case NaryExpr::NaryExprKind::MUL:
    return isSymmetricProduct(this);
  // TODO: implement me.
  case NaryExpr::NaryExprKind::ADD:
    return false;
  }
  llvm_unreachable("Only MUL supported");
}
