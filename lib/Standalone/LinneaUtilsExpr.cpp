//===- LinneaUtilsExpr.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaExpr.h"
#include "llvm/Support/Casting.h"
#include <iostream>

using namespace mlir::linnea::expr;
using namespace llvm;
using namespace std;

static bool isTransposeOfImpl(const Expr *left, const Expr *right) {
  // transpose.
  if (const auto *leftExpr = llvm::dyn_cast_or_null<UnaryExpr>(left))
    if (leftExpr->getKind() == UnaryExpr::UnaryExprKind::TRANSPOSE)
      return leftExpr->getChild() == right;
  if (const auto *rightExpr = llvm::dyn_cast_or_null<UnaryExpr>(right))
    if (rightExpr->getKind() == UnaryExpr::UnaryExprKind::TRANSPOSE)
      return rightExpr->getChild() == left;
  // inverse.
  if (const auto *leftExpr = llvm::dyn_cast_or_null<UnaryExpr>(left))
    if (leftExpr->getKind() == UnaryExpr::UnaryExprKind::INVERSE)
      if (const auto *rightExpr = llvm::dyn_cast_or_null<UnaryExpr>(right))
        if (rightExpr->getKind() == UnaryExpr::UnaryExprKind::INVERSE)
          if (auto *childRightExpr =
                  llvm::dyn_cast_or_null<UnaryExpr>(rightExpr->getChild()))
            if (childRightExpr->getKind() ==
                UnaryExpr::UnaryExprKind::TRANSPOSE)
              return childRightExpr == leftExpr->getChild();
  if (const auto *rightExpr = llvm::dyn_cast_or_null<UnaryExpr>(left))
    if (rightExpr->getKind() == UnaryExpr::UnaryExprKind::INVERSE)
      if (auto *rightExprChild =
              llvm::dyn_cast_or_null<UnaryExpr>(rightExpr->getChild()))
        if (rightExprChild->getKind() == UnaryExpr::UnaryExprKind::TRANSPOSE)
          if (const auto *leftExpr = llvm::dyn_cast_or_null<UnaryExpr>(left))
            if (leftExpr->getKind() == UnaryExpr::UnaryExprKind::TRANSPOSE)
              return rightExprChild == leftExpr->getChild();
  // mul each operand of the left expression should be the transpose
  // of the right expression.
  if (const auto *leftExpr = llvm::dyn_cast_or_null<NaryExpr>(left))
    if (leftExpr->getKind() == NaryExpr::NaryExprKind::MUL)
      if (const auto *rightExpr = llvm::dyn_cast_or_null<NaryExpr>(right))
        if (rightExpr->getKind() == NaryExpr::NaryExprKind::MUL) {
          if (leftExpr->getChildren().size() != rightExpr->getChildren().size())
            return false;
          for (size_t i = 0; i < leftExpr->getChildren().size(); i++)
            if (!leftExpr->getChildren()[i]->isTransposeOf(
                    rightExpr->getChildren()[i]))
              return false;
          return true;
        }
  // both operands. Left and right are the same operand
  // with the symmetric property.
  if (const auto *leftExpr = llvm::dyn_cast_or_null<Operand>(left))
    if (const auto *rightExpr = llvm::dyn_cast_or_null<Operand>(right))
      if (leftExpr->isSymmetric() && leftExpr == rightExpr)
        return true;
  return false;
}

bool Expr::isTransposeOf(const Expr *right) {
  return isTransposeOfImpl(this, right);
}

static bool isSameImpl(const Expr *tree1, const Expr *tree2) {
  if (!tree1 && !tree2)
    return true;

  if (tree1 && tree2) {
    if (tree1->getKind() != tree2->getKind())
      return false;
    // pt comparison for operands.
    if (llvm::isa<Operand>(tree1)) {
      return tree1 == tree2;
    }
    // unary.
    if (llvm::isa<UnaryExpr>(tree1) && llvm::isa<UnaryExpr>(tree2)) {
      const UnaryExpr *tree1Op = llvm::dyn_cast_or_null<UnaryExpr>(tree1);
      const UnaryExpr *tree2Op = llvm::dyn_cast_or_null<UnaryExpr>(tree2);
      // different unaries op.
      if (tree1Op->getKind() != tree2Op->getKind())
        return false;
      return isSameImpl(tree1Op->getChild(), tree2Op->getChild());
    }
    // binary.
    if (llvm::isa<NaryExpr>(tree1) && llvm::isa<NaryExpr>(tree2)) {
      const NaryExpr *tree1Op = llvm::dyn_cast_or_null<NaryExpr>(tree1);
      const NaryExpr *tree2Op = llvm::dyn_cast_or_null<NaryExpr>(tree2);
      // different binary ops.
      if (tree1Op->getKind() != tree2Op->getKind())
        return false;
      // different number of children.
      if (tree1Op->getChildren().size() != tree2Op->getChildren().size())
        return false;
      int numberOfChildren = tree1Op->getChildren().size();
      for (int i = 0; i < numberOfChildren; i++)
        if (!isSameImpl(tree1Op->getChildren()[i], tree2Op->getChildren()[i]))
          return false;
      return true;
    }
  }
  return false;
}

Expr *Operand::getNormalForm() { return this; }

Expr *NaryExpr::getNormalForm() {
  SmallVector<Expr *, 4> operands;
  for (auto *child : this->getChildren())
    operands.push_back(child->getNormalForm());
  return variadicMul(operands, /*fold*/ true);
}

// TODO: avoid duplicating code when problem more specified.
Expr *UnaryExpr::getNormalForm() {
  Expr *child = this->getChild();
  assert(child && "must be valid");
  // normal form transpose.
  if (this->getKind() == UnaryExprKind::TRANSPOSE) {
    // when a mul is a child.
    if (NaryExpr *maybeMul = llvm::dyn_cast_or_null<NaryExpr>(child)) {
      assert(maybeMul->getKind() == NaryExpr::NaryExprKind::MUL);
      SmallVector<Expr *, 4> normalFormOperands;
      auto children = maybeMul->getChildren();
      int size = children.size();
      for (int i = size - 1; i >= 0; i--)
        normalFormOperands.push_back(trans(children.at(i)->getNormalForm()));
      return variadicMul(normalFormOperands, /*fold*/ true);
    }
  }
  // normal form inverse.
  else if (this->getKind() == UnaryExprKind::INVERSE) {
    // when a mul is a child.
    if (NaryExpr *maybeMul = llvm::dyn_cast_or_null<NaryExpr>(child)) {
      assert(maybeMul->getKind() == NaryExpr::NaryExprKind::MUL);
      SmallVector<Expr *, 4> normalFormOperands;
      auto children = maybeMul->getChildren();
      int size = children.size();
      for (int i = size - 1; i >= 0; i--)
        normalFormOperands.push_back(inv(children.at(i)->getNormalForm()));
      return variadicMul(normalFormOperands, /*fold*/ true);
    }
  }
  // normal form operand.
  auto *operand = llvm::dyn_cast_or_null<Operand>(child);
  assert(operand && "expect valid operand");
  assert((this->getKind() == UnaryExprKind::INVERSE) ||
         (this->getKind() == UnaryExprKind::TRANSPOSE));
  if (this->getKind() == UnaryExprKind::INVERSE)
    return inv(child);
  else
    return trans(child);
}

bool Expr::isSame(const Expr *right) {
#if DEBUG
  cout << "this -> \n";
  walk(this);
  cout << "\n";
  cout << "other -> \n";
  walk(right);
#endif
  if (isSameImpl(this, right))
    return true;
  Expr *canonicalForm = this->getNormalForm();
#if DEBUG
  cout << "normal form for this -> \n";
  walk(canonicalForm);
  cout << "\n";
  cout << "other -> \n";
  walk(right);
#endif
  if (isSameImpl(canonicalForm, right))
    return true;
  return false;
}
