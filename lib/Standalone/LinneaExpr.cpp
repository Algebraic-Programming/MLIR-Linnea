//===- LinneaExpr.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaExpr.h"
#include "Standalone/LinneaAttributes.h"
#include "Standalone/LinneaOps.h"
#include "Standalone/LinneaTypes.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <iostream>
#include <limits>

using namespace mlir;
using namespace mlir::linnea::expr;
using namespace mlir::linnea;
using namespace std;

thread_local int ExprBuilder::operandId = 0;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// print an array of expression properties.
static void printProperties(vector<Expr::ExprProperty> properties) {
  for (size_t i = 0, e = properties.size(); i < e; i++) {
    switch (properties[i]) {
    case Expr::ExprProperty::LOWER_TRIANGULAR:
      cout << "LOWER_TRIANGULAR";
      break;
    case Expr::ExprProperty::UPPER_TRIANGULAR:
      cout << "UPPER_TRIANGULAR";
      break;
    case Expr::ExprProperty::SQUARE:
      cout << "SQUARE";
      break;
    case Expr::ExprProperty::SYMMETRIC:
      cout << "SYMMETRIC";
      break;
    case Expr::ExprProperty::FULL_RANK:
      cout << "FULL_RANK";
      break;
    case Expr::ExprProperty::SPD:
      cout << "SPD";
      break;
    case Expr::ExprProperty::FACTORED:
      cout << "FACTORED";
      break;
    case Expr::ExprProperty::GENERAL:
      cout << "GENERAL";
      break;
    default:
      assert(0 && "UNK");
    }
    if (i != e - 1)
      cout << ", ";
  }
}

/// print the shape of the operand.
static void printShape(ArrayRef<int64_t> shape) {
  for (size_t i = 0, e = shape.size(); i < e; i++) {
    cout << shape[i];
    if (i != e - 1)
      cout << ", ";
  }
}

static void printOptimalParens(const vector<vector<long>> &s, size_t i,
                               size_t j, vector<Expr *> operands) {
  if (i == j) {
    cout << " ";
    Operand *operand = nullptr;
    if (auto *unaryOp = llvm::dyn_cast_or_null<UnaryExpr>(operands[i - 1]))
      operand = llvm::dyn_cast_or_null<Operand>(unaryOp->getChild());
    else
      operand = llvm::dyn_cast_or_null<Operand>(operands[i - 1]);
    assert(operand && "must be non null");
    if (llvm::isa<UnaryExpr>(operands[i - 1]))
      cout << "u(" << operand->getName() << ")";
    else
      cout << operand->getName();
    cout << "  ";
  } else {
    cout << "(";
    printOptimalParens(s, i, s[i][j], operands);
    printOptimalParens(s, s[i][j] + 1, j, operands);
    cout << ")";
  }
}

#if DEBUG
static void print(vector<vector<Expr *>> &tmps, bool bitLayout = false) {
  int rows = tmps.size();
  int cols = tmps[0].size();

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (tmps[i][j]) {
        if (bitLayout)
          cout << "1 ";
        else
          walk(tmps[i][j]);
      } else {
        if (bitLayout)
          cout << "0 ";
      }
    }
    cout << "\n";
  }
}
#endif

bool isGEMMLikePattern(NaryExpr *node) {
  Expr *leftExpr = node->getChildren()[0];
  while (auto *unaryExpr = llvm::dyn_cast_or_null<UnaryExpr>(leftExpr)) {
    if (unaryExpr->getKind() == UnaryExpr::UnaryExprKind::INVERSE)
      return false;
    leftExpr = unaryExpr->getChild();
  }
  auto *leftOperand = llvm::dyn_cast_or_null<Operand>(leftExpr);
  if (!leftOperand)
    return false;
  Expr *rightExpr = node->getChildren()[1];
  while (auto *unaryExpr = llvm::dyn_cast_or_null<UnaryExpr>(rightExpr)) {
    if (unaryExpr->getKind() == UnaryExpr::UnaryExprKind::INVERSE)
      return false;
    rightExpr = unaryExpr->getChild();
  }
  auto *rightOperand = llvm::dyn_cast_or_null<Operand>(rightExpr);
  if (!rightOperand)
    return false;
  return leftOperand != rightOperand;
}

bool isGEMM(NaryExpr *node) { return isGEMMLikePattern(node); }

bool isTRMM(NaryExpr *node) {
  return node->getChildren()[0]->isLowerTriangular() && isGEMMLikePattern(node);
}

bool isSYMM(NaryExpr *node) {
  return node->getChildren()[0]->isSymmetric() && isGEMMLikePattern(node);
}

bool isTRSM(NaryExpr *node) { return false; }

bool isSYRK(NaryExpr *node) { return false; }

// Do a simple pattern matching on 'node'. Generalize later if needed.
// Do we want to call BLAS or use Linalg for code-generation?
int getCostBasedOnProperties(Expr *node, int m, int n, int k) {
  auto *binaryExpr = llvm::dyn_cast_or_null<NaryExpr>(node);
  assert(binaryExpr && "must be non null");
  assert(binaryExpr->getChildren().size() == 2 && "expect two children");

  // detect specific cases.
  if (isTRMM(binaryExpr) || isSYMM(binaryExpr) || (isTRSM(binaryExpr)))
    return m * n * k;
  else if (isSYRK(binaryExpr))
    return m * m * k;
  else if (isGEMM(binaryExpr))
    return m * n * k << 1;

  // left child is not a blas pattern but known to be symmetric.
  // or lowertriangular.
  if (binaryExpr->getChildren()[0]->isSymmetric())
    return m * n * k;
  if (binaryExpr->getChildren()[0]->isLowerTriangular())
    return m * n * k;

  return m * n * k << 1;
}

// Compute the cost of the current matrix multiplication based on
// flop count and properties.
pair<long, long> getKernelCostImpl(Expr *node, long &cost, bool fullTree) {
  if (node) {
    if (auto *binaryOp = llvm::dyn_cast_or_null<NaryExpr>(node)) {
      auto children = binaryOp->getChildren();
      assert(children.size() == 2 && "expect only two children");
      pair<long, long> left = getKernelCostImpl(children[0], cost, fullTree);
      pair<long, long> right = getKernelCostImpl(children[1], cost, fullTree);
      // note this cost must be the cost of the top level expr
      // not the cost of the tree.
      auto currentCost =
          getCostBasedOnProperties(node, left.first, left.second, right.second);

      if (fullTree)
        cost += currentCost;
      else
        cost = currentCost;

      return {left.first, right.second};
    }
    if (auto *unaryOp = llvm::dyn_cast_or_null<UnaryExpr>(node)) {
      return getKernelCostImpl(unaryOp->getChild(), cost, fullTree);
    }
    if (auto *operand = llvm::dyn_cast_or_null<Operand>(node)) {
      auto shape = operand->getShape();
      assert(shape.size() == 2 && "must be 2d");
      return {shape[0], shape[1]};
    }
  }
  return {0, 0};
}

void getKernelCostFullExpr(Expr *node, long &cost) {
  (void)getKernelCostImpl(node, cost, true);
}

void getKernelCostTopLevelExpr(Expr *node, long &cost) {
  (void)getKernelCostImpl(node, cost, false);
}

// TODO: we can use the same set of enums between MLIR and the side data
// structure.
std::vector<Expr::ExprProperty>
convert(llvm::ArrayRef<LinneaMatrixEncodingAttr::MatrixProperty> properties) {
  vector<Expr::ExprProperty> result;
  for (auto property : properties) {
    switch (property) {
    case LinneaMatrixEncodingAttr::MatrixProperty::General:
      result.push_back(Expr::ExprProperty::GENERAL);
      break;
    case LinneaMatrixEncodingAttr::MatrixProperty::FullRank:
      result.push_back(Expr::ExprProperty::FULL_RANK);
      break;
    case LinneaMatrixEncodingAttr::MatrixProperty::Diagonal:
      result.push_back(Expr::ExprProperty::DIAGONAL);
      break;
    case LinneaMatrixEncodingAttr::MatrixProperty::UnitDiagonal:
      result.push_back(Expr::ExprProperty::UNIT_DIAGONAL);
      break;
    case LinneaMatrixEncodingAttr::MatrixProperty::LowerTriangular:
      result.push_back(Expr::ExprProperty::LOWER_TRIANGULAR);
      break;
    case LinneaMatrixEncodingAttr::MatrixProperty::UpperTriangular:
      result.push_back(Expr::ExprProperty::UPPER_TRIANGULAR);
      break;
    case LinneaMatrixEncodingAttr::MatrixProperty::Symmetric:
      result.push_back(Expr::ExprProperty::SYMMETRIC);
      break;
    case LinneaMatrixEncodingAttr::MatrixProperty::SPD:
      result.push_back(Expr::ExprProperty::SPD);
      break;
    case LinneaMatrixEncodingAttr::MatrixProperty::SPSD:
      result.push_back(Expr::ExprProperty::SPSD);
      break;
    case LinneaMatrixEncodingAttr::MatrixProperty::Square:
      result.push_back(Expr::ExprProperty::SQUARE);
      break;
    case LinneaMatrixEncodingAttr::MatrixProperty::Factored:
      result.push_back(Expr::ExprProperty::FACTORED);
      break;
    }
  }
  return result;
}

llvm::SmallVector<LinneaMatrixEncodingAttr::MatrixProperty>
convert(std::vector<Expr::ExprProperty> properties) {
  llvm::SmallVector<LinneaMatrixEncodingAttr::MatrixProperty> result;
  for (auto property : properties) {
    switch (property) {
    case Expr::ExprProperty::GENERAL:
      result.push_back(LinneaMatrixEncodingAttr::MatrixProperty::General);
      break;
    case Expr::ExprProperty::FULL_RANK:
      result.push_back(LinneaMatrixEncodingAttr::MatrixProperty::FullRank);
      break;
    case Expr::ExprProperty::DIAGONAL:
      result.push_back(LinneaMatrixEncodingAttr::MatrixProperty::Diagonal);
      break;
    case Expr::ExprProperty::UNIT_DIAGONAL:
      result.push_back(LinneaMatrixEncodingAttr::MatrixProperty::UnitDiagonal);
      break;
    case Expr::ExprProperty::LOWER_TRIANGULAR:
      result.push_back(
          LinneaMatrixEncodingAttr::MatrixProperty::LowerTriangular);
      break;
    case Expr::ExprProperty::UPPER_TRIANGULAR:
      result.push_back(
          LinneaMatrixEncodingAttr::MatrixProperty::UpperTriangular);
      break;
    case Expr::ExprProperty::SYMMETRIC:
      result.push_back(LinneaMatrixEncodingAttr::MatrixProperty::Symmetric);
      break;
    case Expr::ExprProperty::SPD:
      result.push_back(LinneaMatrixEncodingAttr::MatrixProperty::SPD);
      break;
    case Expr::ExprProperty::SPSD:
      result.push_back(LinneaMatrixEncodingAttr::MatrixProperty::SPSD);
      break;
    case Expr::ExprProperty::SQUARE:
      result.push_back(LinneaMatrixEncodingAttr::MatrixProperty::Square);
      break;
    case Expr::ExprProperty::FACTORED:
      result.push_back(LinneaMatrixEncodingAttr::MatrixProperty::Factored);
      break;
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// ExprBuilder
//===----------------------------------------------------------------------===//

Expr *ExprBuilder::buildOperandImpl(Value val) {

  if (contains(val))
    return lookup(val);

  Expr *operand = nullptr;
  if (auto matrixType = val.getType().dyn_cast_or_null<MatrixType>()) {
    auto properties = convert(matrixType.getProperty()
                                  .cast<LinneaMatrixEncodingAttr>()
                                  .getEncoding());
    std::string id = "A" + std::to_string(getNextId());
    // FIXME: this is a terrible hack!
    // MatrixType returns an ArrayRef but the constructor
    // of Matrix or Identity takes a SmallVector.
    vector<int64_t> dims = matrixType.getDims().vec();
    SmallVector<int64_t, 2> dimsVec = {dims.begin(), dims.end()};
    operand = new Matrix(id, dimsVec);
    operand->setProperties(properties);
  } else if (auto identityType =
                 val.getType().dyn_cast_or_null<IdentityType>()) {
    vector<int64_t> dims = identityType.getDims().vec();
    SmallVector<int64_t, 2> dimsVec = {dims.begin(), dims.end()};
    operand = new Identity(dimsVec);
  } else {
    llvm_unreachable("expect either a MatrixType or an IdentityType");
  }

  assert(operand && "must be non null");
  // map in both directions.
  map(val, operand);
  map(operand, val);
  return operand;
}

static Type getElementType(Type t) {
  if (auto mt = t.dyn_cast_or_null<MatrixType>())
    return mt.getElementType();
  if (auto it = t.dyn_cast_or_null<IdentityType>())
    return it.getElementType();
  llvm_unreachable("expect only MatrixType or IdentityType");
}

static int64_t getDimSizeAtPos(Type t, size_t pos) {
  if (auto mt = t.dyn_cast_or_null<MatrixType>()) {
    assert(pos < mt.getDims().size());
    return mt.getDims()[pos];
  }
  if (auto it = t.dyn_cast_or_null<IdentityType>()) {
    assert(pos < it.getDims().size());
    return it.getDims()[pos];
  }
  llvm_unreachable("expect only MatrixType or IdentityType");
}

static StringAttr getStringAttributeFromEnum(NaryExpr::SemiringsKind kind,
                                             OpBuilder &builder) {
  if (kind == NaryExpr::SemiringsKind::REAL_ARITH)
    return StringAttr::get(builder.getContext(), "real-arith");
  else if (kind == NaryExpr::SemiringsKind::INTEGER_ARITH)
    return StringAttr::get(builder.getContext(), "integer-arith");
  else if (kind == NaryExpr::SemiringsKind::MIN_PLUS)
    return StringAttr::get(builder.getContext(), "min-plus");
  else if (kind == NaryExpr::SemiringsKind::MAX_PLUS)
    return StringAttr::get(builder.getContext(), "max-plus");
  llvm_unreachable("unknown semiring");
}

template <typename OP>
Value ExprBuilder::buildBinaryOpImpl(Location loc, OpBuilder &builder,
                                     NaryExpr *expr) {
  SmallVector<Value> operands;
  auto children = expr->getChildren();
  // at this point the NaryExpr is expected to have 2 children.
  assert(children.size() == 2 && "expect two children");
  for (int i = 0, e = children.size(); i < e; i++)
    operands.push_back(buildIRImpl(loc, builder, children[i]));
  SmallVector<LinneaMatrixEncodingAttr::MatrixProperty> properties =
      convert(expr->getAndSetProperties());
  Type elementType = getElementType(operands[0].getType());
  SmallVector<int64_t> dims = {
      getDimSizeAtPos(operands[0].getType(), 0),
      getDimSizeAtPos(operands[operands.size() - 1].getType(), 1)};
  MatrixType result = MatrixType::get(
      builder.getContext(),
      LinneaMatrixEncodingAttr::get(builder.getContext(), properties), dims,
      elementType);
  assert(operands.size() == 2 && "expect two operands");
  StringAttr semirings =
      getStringAttributeFromEnum(expr->getSemiringsKind(), builder);
  return builder.create<OP>(loc, result, operands[0], operands[1], semirings);
}

template <typename OP>
Value ExprBuilder::buildUnaryOpImpl(Location loc, OpBuilder &builder,
                                    UnaryExpr *expr) {
  Value operand = buildIRImpl(loc, builder, expr->getChild());
  return builder.create<OP>(loc, operand.getType(), operand);
}

Value ExprBuilder::buildIRImpl(Location loc, OpBuilder &builder, Expr *root) {
  if (root) {
    if (auto *naryExpr = llvm::dyn_cast_or_null<NaryExpr>(root)) {
      switch (naryExpr->getKind()) {
      case NaryExpr::NaryExprKind::MUL:
        return buildBinaryOpImpl<MulOpLow>(loc, builder, naryExpr);
        break;
      case NaryExpr::NaryExprKind::ADD:
        return buildBinaryOpImpl<AddOpLow>(loc, builder, naryExpr);
        break;
      }
    }
    if (auto *unaryExpr = llvm::dyn_cast_or_null<UnaryExpr>(root)) {
      switch (unaryExpr->getKind()) {
      case UnaryExpr::UnaryExprKind::TRANSPOSE:
        return buildUnaryOpImpl<TransposeOp>(loc, builder, unaryExpr);
        break;
      case UnaryExpr::UnaryExprKind::INVERSE:
        return buildUnaryOpImpl<InverseOpLow>(loc, builder, unaryExpr);
        break;
      }
    }
    if (auto *operand = llvm::dyn_cast_or_null<Operand>(root)) {
      return lookup(operand);
    }
  }
  llvm_unreachable("Expression not supported!");
  return nullptr;
}

Value ExprBuilder::buildIR(Location loc, OpBuilder &builder, Expr *root) {
  return buildIRImpl(loc, builder, root);
}

// The user of source should be a child of target, which is
// another equationOp.
static bool hasOnlyUser(Value source, Operation *target) {
  for (Operation *user : source.getUsers()) {
    if (user->getParentOp() != target)
      return false;
  }
  return true;
}

static inline NaryExpr::SemiringsKind getSemiringsKindFromOp(Operation *op) {
  assert(
      (op->getName().getIdentifier().str().compare("linnea.mul.high") == 0) ||
      (op->getName().getIdentifier().str().compare("linnea.add.high") == 0));
  StringAttr attr = op->getAttr("semirings").dyn_cast_or_null<StringAttr>();
  assert(attr);
  if (attr.str().compare("real-arith") == 0)
    return NaryExpr::SemiringsKind::REAL_ARITH;
  else if (attr.str().compare("integer-arith") == 0)
    return NaryExpr::SemiringsKind::INTEGER_ARITH;
  else if (attr.str().compare("min-plus") == 0)
    return NaryExpr::SemiringsKind::MIN_PLUS;
  else if (attr.str().compare("max-plus") == 0)
    return NaryExpr::SemiringsKind::MAX_PLUS;
  llvm_unreachable("unknown semirings");
}

// Build symbolic expression following use-def chain for 'val'.
Expr *ExprBuilder::buildExprImpl(Value val, Operation *currentOp) {
  // 'val' comes is a basic block arg or the result
  // of a fillOp. Build operand directly.
  if (auto blockArg = val.dyn_cast_or_null<BlockArgument>()) {
    return buildOperandImpl(blockArg);
  }
  Operation *defOp = val.getDefiningOp();
  assert(defOp && "must be valid");
  if (auto fillOp = dyn_cast_or_null<linnea::FillOp>(defOp)) {
    return buildOperandImpl(fillOp.result());
  }
  // 'val' is the result of another linnea operations, recurse
  // untill we get a basic block arg or a result of a fillOp.
  if (auto mulOp = dyn_cast_or_null<linnea::MulOpHigh>(defOp)) {
    SmallVector<Expr *, 4> children;
    for (Value operand : mulOp.getOperands())
      children.push_back(buildExprImpl(operand, currentOp));
    return variadicMul(children, /*fold*/ true, getSemiringsKindFromOp(mulOp));
  }
  if (auto addOp = dyn_cast_or_null<linnea::AddOpHigh>(defOp)) {
    SmallVector<Expr *, 4> children;
    for (Value operand : addOp.getOperands())
      children.push_back(buildExprImpl(operand, currentOp));
    return variadicAdd(children, /*fold*/ true, getSemiringsKindFromOp(addOp));
  }
  if (auto transOp = dyn_cast_or_null<linnea::TransposeOp>(defOp)) {
    Expr *child = buildExprImpl(transOp.getOperand(), currentOp);
    return trans(child);
  }
  if (auto invOp = dyn_cast_or_null<linnea::InverseOpHigh>(defOp)) {
    Expr *child = buildExprImpl(invOp.getOperand(), currentOp);
    return inv(child);
  }
  if (auto eqOp = dyn_cast_or_null<linnea::EquationOp>(defOp)) {
    // fold the eqOp in currentOp as eqOp has currentOp
    // as only user.
    if (hasOnlyUser(eqOp.getResult(), currentOp)) {
      Block &block = eqOp.getBody();
      Operation *terminator = block.getTerminator();
      visited.insert(eqOp.getOperation());
      return buildExprImpl(terminator->getOperand(0), currentOp);
    }
    // current equation op (eqOp) is used by the 'currentOp' but it has
    // multiple users. Build it and use the newly created value as operand for
    // the 'currentOp'.
    Block &block = eqOp.getBody();
    Operation *terminator = block.getTerminator();
    Value termOperand = terminator->getOperand(0);
    Expr *root = buildLinneaExpr(termOperand, eqOp.getOperation());
    // FIXME: see ExprBuilder(bool)
    root = root->simplify(symbolicOpt);
    OpBuilder builder(eqOp->getContext());
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(eqOp);
    Value rootVal = buildIR(eqOp->getLoc(), builder, root);
    Value resultEqOp = eqOp.getResult();
    resultEqOp.replaceAllUsesWith(rootVal);

    // insert to visited to generate it only once.
    visited.insert(eqOp.getOperation());
    return buildOperandImpl(rootVal);
  }
  llvm_unreachable("operation not handled");
  return nullptr;
}

Expr *ExprBuilder::buildLinneaExpr(Value val, Operation *op) {
  return buildExprImpl(val, op);
}

//===----------------------------------------------------------------------===//
// Expr
//===----------------------------------------------------------------------===//

/// check if `expr` has `property`.
template <typename T>
bool Expr::hasProperty(T *expr, Expr::ExprProperty property) {
  return (expr->inferredProperties).count(property) != 0;
}

/// Query each property on `expr`.
// TODO: find a better way to have a class of properties.
// (i.e., triangular implies square but you may also want to have multiple types
// of triangular matrices). Passing all the properties to later conversions is
// annoying.
template <typename T>
void Expr::setPropertiesImpl(T *expr) {
  // if the lookup fails check if the expr has the given property.
  if (!hasProperty<T>(expr, Expr::ExprProperty::GENERAL) && expr->isGeneral())
    expr->inferredProperties.insert(Expr::ExprProperty::GENERAL);
  if (!hasProperty<T>(expr, Expr::ExprProperty::UPPER_TRIANGULAR) &&
      expr->isUpperTriangular())
    expr->inferredProperties.insert(Expr::ExprProperty::UPPER_TRIANGULAR);
  if (!hasProperty<T>(expr, Expr::ExprProperty::UPPER_TRIANGULAR) &&
      expr->isLowerTriangular())
    expr->inferredProperties.insert(Expr::ExprProperty::LOWER_TRIANGULAR);
  // if (!hasProperty<T>(expr, Expr::ExprProperty::SQUARE) && expr->isSquare())
  //  expr->inferredProperties.insert(Expr::ExprProperty::SQUARE);
  if (!hasProperty<T>(expr, Expr::ExprProperty::SYMMETRIC) &&
      expr->isSymmetric())
    expr->inferredProperties.insert(Expr::ExprProperty::SYMMETRIC);
  // if (!hasProperty<T>(expr, Expr::ExprProperty::FULL_RANK) &&
  //    expr->isFullRank())
  //  expr->inferredProperties.insert(Expr::ExprProperty::FULL_RANK);
  if (!hasProperty<T>(expr, Expr::ExprProperty::SPD) && expr->isSPD())
    expr->inferredProperties.insert(Expr::ExprProperty::SPD);
}

#define LEVEL_SPACES 2

/// Walk a generic expression.
void Expr::walk(int level) const {
  if (const auto *naryExpr = llvm::dyn_cast_or_null<NaryExpr>(this)) {
    switch (naryExpr->getKind()) {
    case NaryExpr::NaryExprKind::MUL:
      cout << string(level, ' ') << "(*\n";
      break;
    case NaryExpr::NaryExprKind::ADD:
      cout << string(level, ' ') << "(+\n";
      break;
    }
    for (const Expr *child : naryExpr->getChildren()) {
      child->walk(level + LEVEL_SPACES);
      cout << " \n";
    }
  } // binaryOp
  if (const auto *unaryOp = llvm::dyn_cast_or_null<UnaryExpr>(this)) {
    switch (unaryOp->getKind()) {
    case UnaryExpr::UnaryExprKind::TRANSPOSE:
      cout << string(level, ' ') << "transpose(";
      break;
    case UnaryExpr::UnaryExprKind::INVERSE:
      cout << string(level, ' ') << "inverse(";
      break;
    }
    unaryOp->getChild()->walk(level);
    cout << string(level, ' ') << ")";
  } // unaryOp
  if (const auto *operand = llvm::dyn_cast_or_null<Operand>(this)) {
    cout << string(level, ' ') << operand->getName() << " [";
    printProperties(operand->getProperties());
    cout << "] [";
    printShape(operand->getShape());
    cout << "]";
  } // operand
}

static SmallVector<long, 8> getOperandSize(ArrayRef<Expr *> operands) {
  SmallVector<long, 8> sizes;
  for (Expr *operand : operands) {
    SmallVector<int64_t, 2> shape = operand->getResultShape();
    if (!sizes.size()) {
      sizes.push_back(shape[0]);
      sizes.push_back(shape[1]);
    } else
      sizes.push_back(shape[1]);
  }
  return sizes;
}

struct ResultMCO {
  Expr *optimizedMulExpr = nullptr;
  long flops = 0;
};

// Run matrix-chain optimization.
static ResultMCO runMCO(ArrayRef<Expr *> operands,
                        NaryExpr::SemiringsKind semirings) {

  SmallVector<long, 8> pVector = getOperandSize(operands);
  const size_t n = pVector.size();
  vector<vector<long>> m(n, vector<long>(n, std::numeric_limits<long>::max()));
  vector<vector<long>> s(n, vector<long>(n, std::numeric_limits<long>::max()));
  // store symbolic temporary variables representing sub-chains.
  vector<vector<Expr *>> tmps(n, vector<Expr *>(n, nullptr));

  for (size_t i = 0; i < n - 1; i++)
    tmps[i + 1][i + 1] = operands[i];

#if DEBUG
  cout << "\n\n-before-tmps-\n";
  print(tmps, true);
#endif

  for (size_t i = 0; i < n; i++)
    m[i][i] = 0;

  size_t j = 0;
  long q = 0;
  for (size_t l = 2; l < n; l++) {
    for (size_t i = 1; i < n - l + 1; i++) {
      j = i + l - 1;
      m[i][j] = std::numeric_limits<long>::max();
      for (size_t k = i; k <= j - 1; k++) {

        auto *tmpexpr = variadicMul({tmps[i][k], tmps[k + 1][j]},
                                    /*fold*/ false, semirings);
#if DEBUG
        cout << "---\n";
        walk(tmpexpr);
        cout << "\n---\n\n";
#endif
        long cost = 0;
        getKernelCostTopLevelExpr(tmpexpr, cost);
        q = m[i][k] + m[k + 1][j] + cost;
        if (q < m[i][j]) {
          tmps[i][j] = variadicMul({tmps[i][k], tmps[k + 1][j]}, /*fold*/ false,
                                   semirings);
          m[i][j] = q;
          s[i][j] = k;
        }
      }
    }
  }

#if DEBUG
  cout << "\n\n-after-tmps-\n";
  print(tmps, true);
  cout << "\n";
  walk(tmps[1][tmps.size() - 1]);

  cout << "\n\n-----s------\n";
  int rows = s.size();
  int cols = s[0].size();
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (s[i][j] == std::numeric_limits<long>::max())
        cout << "- ";
      else
        cout << s[i][j] << " ";
    }
    cout << "\n";
  }
  cout << "\n-----m------\n";
  rows = m.size();
  cols = m[0].size();
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (m[i][j] == std::numeric_limits<long>::max())
        cout << "- ";
      else
        cout << m[i][j] << " ";
    }
    cout << "\n";
  }
  cout << "\n";
  printOptimalParens(s, 1, operands.size(), operands);
  cout << "\n\n";
#endif
  return {tmps[1][tmps.size() - 1], m[1][m.size() - 1]};
}

// preserve the chain order.
static Expr *passThroughMul(ArrayRef<Expr *> operands,
                            NaryExpr::SemiringsKind semirings) {
  assert(operands.size() >= 2 && "expect two or more operands");
  Expr *root =
      variadicMul({operands[0], operands[1]}, /*fold*/ false, semirings);
  for (size_t i = 2; i < operands.size(); i++)
    root = variadicMul({root, operands[i]}, /*fold*/ false, semirings);
  return root;
}

static Expr *passThroughAdd(ArrayRef<Expr *> operands,
                            NaryExpr::SemiringsKind semirings) {
  assert(operands.size() >= 2 && "expect two or more operands");
  Expr *root =
      variadicAdd({operands[0], operands[1]}, /*fold*/ false, semirings);
  for (size_t i = 2; i < operands.size(); i++)
    root = variadicAdd({root, operands[i]}, /*fold*/ false, semirings);
  return root;
}

// Optimize symbolic expression.
// symbolicOpt flag drives the optimization. Currenty
// we do only MC. Disable it if symbolicOpt = false.
static Expr *simplifyImpl(bool symbolicOpt, Expr *root) {
  if (root) {
    if (auto *naryExpr = llvm::dyn_cast_or_null<NaryExpr>(root)) {
      auto children = naryExpr->getChildren();
      assert(children.size() >= 2 && "expect two or more children");
      // Optimize each children then run MCO.
      SmallVector<Expr *> simplifiedChildren;
      for (size_t i = 0; i < children.size(); i++)
        simplifiedChildren.push_back(simplifyImpl(symbolicOpt, children[i]));
      switch (naryExpr->getKind()) {
      case NaryExpr::NaryExprKind::MUL: {
        if (symbolicOpt)
          return runMCO(simplifiedChildren, naryExpr->getSemiringsKind())
              .optimizedMulExpr;
        return passThroughMul(simplifiedChildren, naryExpr->getSemiringsKind());
      }
      case NaryExpr::NaryExprKind::ADD: {
        return passThroughAdd(simplifiedChildren, naryExpr->getSemiringsKind());
      }
      }
    }
    if (auto *unaryExpr = llvm::dyn_cast_or_null<UnaryExpr>(root)) {
      auto *child = simplifyImpl(symbolicOpt, unaryExpr->getChild());
      switch (unaryExpr->getKind()) {
      case UnaryExpr::UnaryExprKind::TRANSPOSE:
        return trans(child);
      case UnaryExpr::UnaryExprKind::INVERSE:
        return inv(child);
      }
    }
    if (auto *operand = llvm::dyn_cast_or_null<Operand>(root))
      return operand;
  }
  llvm_unreachable("expect root to be non null");
}

Expr *Expr::simplify(bool symbolicOpt) {
  return simplifyImpl(symbolicOpt, this);
}

//===----------------------------------------------------------------------===//
// UnaryExpr
//===----------------------------------------------------------------------===//

/// Return a vector of inferred properties by calling `setPropertiesImpl`.
vector<Expr::ExprProperty> UnaryExpr::getAndSetProperties() {
  vector<Expr::ExprProperty> inferredPropertiesAsVec;
  setPropertiesImpl<UnaryExpr>(this);
  for (auto property : inferredProperties)
    inferredPropertiesAsVec.push_back(property);
  return inferredPropertiesAsVec;
}

/// Return the size of the unary expr operand.
SmallVector<int64_t, 2> UnaryExpr::getResultShape() const {
  if (this->getKind() == UnaryExprKind::INVERSE)
    return this->getChild()->getResultShape();
  else { // transpose.
    assert(this->getKind() == UnaryExprKind::TRANSPOSE);
    SmallVector<int64_t, 2> dims = this->getChild()->getResultShape();
    assert(dims.size() == 2 && "expect two dimensions");
    std::swap(dims[0], dims[1]);
    return dims;
  }
}

//===----------------------------------------------------------------------===//
// NaryExpr
//===----------------------------------------------------------------------===//

/// Return a vector of inferred properties by calling `setPropertiesImpl`.
vector<Expr::ExprProperty> NaryExpr::getAndSetProperties() {
  vector<Expr::ExprProperty> inferredPropertiesAsVec;
  setPropertiesImpl<NaryExpr>(this);
  for (auto property : inferredProperties)
    inferredPropertiesAsVec.push_back(property);
  return inferredPropertiesAsVec;
}

/// Return the size of the nary expr operand.
SmallVector<int64_t, 2> NaryExpr::getResultShape() const {
  if (this->getKind() == NaryExprKind::MUL) {
    auto children = this->getChildren();
    assert(children.size() >= 2 && "two or more children expcted");
    int64_t leftDim = children[0]->getResultShape()[0];
    int64_t rightDim = children[children.size() - 1]->getResultShape()[1];
    return {leftDim, rightDim};
  } else {
    assert(this->getKind() == NaryExprKind::ADD);
    // A matrix can only be added to (or subtracted from) another matrix if the
    // two matrices have the same dimensions. Thus return the dimension of the
    // first children.
    return children[0]->getResultShape();
  }
}

/// Return a flop estimate for the matrix-chain algorithm. Used for test only.
// TODO: maybe runMCO should be a method of the class NaryExpr::MUL only.
long NaryExpr::getMCPFlops() {
  assert(this->getKind() == NaryExprKind::MUL);
  auto optimizedMulExpr = runMCO(this->getChildren(), this->getSemiringsKind());
  return optimizedMulExpr.flops;
}

//===----------------------------------------------------------------------===//
// ScopedContext
//===----------------------------------------------------------------------===//

ScopedContext *&ScopedContext::getCurrentScopedContext() {
  thread_local ScopedContext *context = nullptr;
  return context;
}

ScopedContext::~ScopedContext() {
  for (auto *expr : liveRefs)
    delete expr;
}

void ScopedContext::print() {
  cout << "#live refs: " << liveRefs.size() << "\n";
  int operands = 0;
  int unaries = 0;
  int binaries = 0;
  for (Expr *expr : liveRefs) {
    if (llvm::isa<Operand>(expr))
      operands++;
    if (llvm::isa<UnaryExpr>(expr))
      unaries++;
    if (llvm::isa<NaryExpr>(expr))
      binaries++;
  }
  cout << "#operands : " << operands << "\n";
  cout << "#unaries : " << unaries << "\n";
  cout << "#binaries : " << binaries << "\n";
}

//===----------------------------------------------------------------------===//
// Operand
//===----------------------------------------------------------------------===//

/// Check if the shape is square.
bool hasSquareShape(ArrayRef<int64_t> shape) {
  assert(shape.size() >= 1 && "must be >= 1");
  if (shape.size() == 1)
    return true;
  return all_of(shape.begin(), shape.end(),
                [&](int dim) { return dim == shape[0]; });
}

Operand::Operand(string name, SmallVector<int64_t, 2> shape, OperandKind kind)
    : ScopedExpr(ExprKind::OPERAND), name(name), shape(shape), kind(kind) {
  if (hasSquareShape(shape))
    this->setProperties({Expr::ExprProperty::SQUARE});
}

/// Set properties for the current operand.
void Operand::setProperties(std::vector<Expr::ExprProperty> properties) {
  for (auto property : properties) {
    if (property == Expr::ExprProperty::LOWER_TRIANGULAR && !this->isSquare()) {
      llvm::errs()
          << "A triangular matrix is a special kind of square matrix\n";
      continue;
    }
    if (property == Expr::ExprProperty::UPPER_TRIANGULAR && !this->isSquare()) {
      llvm::errs()
          << "A triangular matrix is a special kind of square matrix\n";
      continue;
    }
    if (property == Expr::ExprProperty::SYMMETRIC && !this->isSquare()) {
      llvm::errs() << "A symmetric matrix is a special kind of square matrix\n";
      continue;
    }
    inferredProperties.insert(property);
  }
}

/// Get the shape of the operand.
SmallVector<int64_t, 2> Operand::getResultShape() const { return shape; }

/// Return a vector of properties for the current operand.
vector<Expr::ExprProperty> Operand::getProperties() const {
  vector<Expr::ExprProperty> inferredPropertiesAsVec;
  for (auto property : inferredProperties)
    inferredPropertiesAsVec.push_back(property);
  return inferredPropertiesAsVec;
}

//===----------------------------------------------------------------------===//
// Matrix
//===----------------------------------------------------------------------===//

Matrix::Matrix(string name, SmallVector<int64_t, 2> shape, MatrixKind kind)
    : Operand(name, shape, OperandKind::MATRIX), kind(kind) {
  assert(shape.size() == 2 && "expect shape of size 2");
}

//===----------------------------------------------------------------------===//
// Identity
//===----------------------------------------------------------------------===//

Identity::Identity(SmallVector<int64_t, 2> shape)
    : Matrix("I", shape, MatrixKind::IDENTITY) {}
