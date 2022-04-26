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
