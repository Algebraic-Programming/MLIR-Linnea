//===- LinneaExpr.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/LinneaExpr.h"
#include "Standalone/LinneaOps.h"
#include "Standalone/LinneaTypes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <iostream>
#include <limits>

using namespace mlir::linnea::expr;
using namespace mlir::linnea;
using namespace std;

thread_local int ExprBuilder::operandId = 0;

// TODO: we can use the same set of enums instead of converting each time.
std::vector<Expr::ExprProperty>
convert(llvm::ArrayRef<MatrixType::MatrixProperty> properties) {
  vector<Expr::ExprProperty> result;
  for (auto property : properties) {
    switch (property) {
    case MatrixType::MatrixProperty::General:
      result.push_back(Expr::ExprProperty::GENERAL);
      break;
    case MatrixType::MatrixProperty::FullRank:
      result.push_back(Expr::ExprProperty::FULL_RANK);
      break;
    case MatrixType::MatrixProperty::Diagonal:
      result.push_back(Expr::ExprProperty::DIAGONAL);
      break;
    case MatrixType::MatrixProperty::UnitDiagonal:
      result.push_back(Expr::ExprProperty::UNIT_DIAGONAL);
      break;
    case MatrixType::MatrixProperty::LowerTriangular:
      result.push_back(Expr::ExprProperty::LOWER_TRIANGULAR);
      break;
    case MatrixType::MatrixProperty::UpperTriangular:
      result.push_back(Expr::ExprProperty::UPPER_TRIANGULAR);
      break;
    case MatrixType::MatrixProperty::Symmetric:
      result.push_back(Expr::ExprProperty::SYMMETRIC);
      break;
    case MatrixType::MatrixProperty::SPD:
      result.push_back(Expr::ExprProperty::SPD);
      break;
    case MatrixType::MatrixProperty::SPSD:
      result.push_back(Expr::ExprProperty::SPSD);
      break;
    case MatrixType::MatrixProperty::Identity:
      result.push_back(Expr::ExprProperty::IDENTITY);
      break;
    case MatrixType::MatrixProperty::Square:
      result.push_back(Expr::ExprProperty::SQUARE);
      break;
    case MatrixType::MatrixProperty::Factored:
      result.push_back(Expr::ExprProperty::FACTORED);
      break;
    }
  }
  return result;
}

llvm::SmallVector<MatrixType::MatrixProperty>
convert(std::vector<Expr::ExprProperty> properties) {
  llvm::SmallVector<MatrixType::MatrixProperty> result;
  for (auto property : properties) {
    switch (property) {
    case Expr::ExprProperty::GENERAL:
      result.push_back(MatrixType::MatrixProperty::General);
      break;
    case Expr::ExprProperty::FULL_RANK:
      result.push_back(MatrixType::MatrixProperty::FullRank);
      break;
    case Expr::ExprProperty::DIAGONAL:
      result.push_back(MatrixType::MatrixProperty::Diagonal);
      break;
    case Expr::ExprProperty::UNIT_DIAGONAL:
      result.push_back(MatrixType::MatrixProperty::UnitDiagonal);
      break;
    case Expr::ExprProperty::LOWER_TRIANGULAR:
      result.push_back(MatrixType::MatrixProperty::LowerTriangular);
      break;
    case Expr::ExprProperty::UPPER_TRIANGULAR:
      result.push_back(MatrixType::MatrixProperty::UpperTriangular);
      break;
    case Expr::ExprProperty::SYMMETRIC:
      result.push_back(MatrixType::MatrixProperty::Symmetric);
      break;
    case Expr::ExprProperty::SPD:
      result.push_back(MatrixType::MatrixProperty::SPD);
      break;
    case Expr::ExprProperty::SPSD:
      result.push_back(MatrixType::MatrixProperty::SPSD);
      break;
    case Expr::ExprProperty::IDENTITY:
      result.push_back(MatrixType::MatrixProperty::Identity);
      break;
    case Expr::ExprProperty::SQUARE:
      result.push_back(MatrixType::MatrixProperty::Square);
      break;
    case Expr::ExprProperty::FACTORED:
      result.push_back(MatrixType::MatrixProperty::Factored);
      break;
    }
  }
  return result;
}

Expr *ExprBuilder::buildOperandImpl(Value val) {
  assert(val.getType().cast<MatrixType>() && "expect matrixType");
  if (contains(val))
    return lookup(val);

  auto matrixType = val.getType().cast<MatrixType>();
  auto properties = convert(matrixType.getProperty());
  auto size = matrixType.getDims();
  std::string id = "A" + std::to_string(getNextId());
  Expr *operand = new Operand(id, size);
  operand->setProperties(properties);
  // map in both directions
  map(val, operand);
  map(operand, val);
  return operand;
}

mlir::Value ExprBuilder::buildMulImpl(Location loc, OpBuilder &builder,
                                      NaryExpr *expr,
                                      BlockAndValueMapping &mapper) {
  SmallVector<Value> operands;
  auto children = expr->getChildren();
  for (int i = 0, e = children.size(); i < e; i++)
    operands.push_back(buildIRImpl(loc, builder, children[i], mapper));
  MatrixType first = operands[0].getType().cast<MatrixType>();
  MatrixType last = operands[operands.size() - 1].getType().cast<MatrixType>();
  SmallVector<MatrixType::MatrixProperty> properties =
      convert(expr->getAndSetProperties());
  SmallVector<int64_t> dims = {first.getDims()[0], last.getDims()[1]};
  MatrixType result = MatrixType::get(builder.getContext(), properties, dims,
                                      first.getElementType());
  return builder.create<MulOp>(loc, result, operands);
}

mlir::Value ExprBuilder::buildTransposeImpl(Location loc, OpBuilder &builder,
                                            UnaryExpr *expr,
                                            BlockAndValueMapping &mapper) {
  return nullptr;
}

mlir::Value ExprBuilder::buildInverseImpl(Location loc, OpBuilder &builder,
                                          UnaryExpr *expr,
                                          BlockAndValueMapping &mapper) {
  return nullptr;
}

mlir::Value ExprBuilder::buildIRImpl(Location loc, OpBuilder &builder,
                                     Expr *root, BlockAndValueMapping &mapper) {
  if (root) {
    if (auto naryExpr = llvm::dyn_cast_or_null<NaryExpr>(root)) {
      switch (naryExpr->getKind()) {
      case NaryExpr::NaryExprKind::MUL:
        return buildMulImpl(loc, builder, naryExpr, mapper);
        break;
      default:
        assert(0 && "UNK");
      }
    }
    if (auto unaryExpr = llvm::dyn_cast_or_null<UnaryExpr>(root)) {
      switch (unaryExpr->getKind()) {
      case UnaryExpr::UnaryExprKind::TRANSPOSE:
        return buildTransposeImpl(loc, builder, unaryExpr, mapper);
        break;
      case UnaryExpr::UnaryExprKind::INVERSE:
        return buildInverseImpl(loc, builder, unaryExpr, mapper);
        break;
      default:
        assert(0 && "UNK");
      }
    }
    if (auto operand = llvm::dyn_cast_or_null<Operand>(root)) {
      return mapper.lookup(exprMap[operand]);
    }
  }
  assert(0 && "UNKN");
  return nullptr;
}

mlir::Value ExprBuilder::buildIR(Location loc, OpBuilder &builder, Expr *root,
                                 BlockAndValueMapping &mapper) {
  return buildIRImpl(loc, builder, root, mapper);
}

Expr *ExprBuilder::buildExprImpl(Value val) {
  if (auto blockArg = val.dyn_cast_or_null<BlockArgument>()) {
    return buildOperandImpl(blockArg);
  }

  Operation *defOp = val.getDefiningOp();
  assert(defOp && "must be valid");
  if (auto mulOp = dyn_cast_or_null<mlir::linnea::MulOp>(defOp)) {
    std::vector<Expr *> children;
    for (Value operand : mulOp.getOperands()) {
      children.push_back(buildExprImpl(operand));
    }
    return variadicMul(children);
  }
  if (auto transOp = dyn_cast_or_null<mlir::linnea::TransposeOp>(defOp)) {
    Expr *child = buildExprImpl(transOp.getOperand());
    return trans(child);
  }
  if (auto invOp = dyn_cast_or_null<mlir::linnea::InverseOp>(defOp)) {
    Expr *child = buildExprImpl(invOp.getOperand());
    return inv(child);
  }
  assert(0 && "unreachable");
  return nullptr;
}

Expr *ExprBuilder::buildLinneaExpr(Value val) { return buildExprImpl(val); }

vector<int64_t> Operand::getResultDimensions() const { return shape; }

vector<int64_t> UnaryExpr::getResultDimensions() const {
  if (this->getKind() == UnaryExprKind::INVERSE)
    return this->getChild()->getResultDimensions();
  else {
    vector<int64_t> dims = this->getChild()->getResultDimensions();
    assert(dims.size() == 2 && "expect two dimensions");
    std::swap(dims[0], dims[1]);
    return dims;
  }
  assert(0 && "unreachable");
  return {};
}

vector<int64_t> NaryExpr::getResultDimensions() const {
  if (this->getKind() == NaryExprKind::MUL) {
    auto children = this->getChildren();
    assert(children.size() >= 2 && "two or more children expcted");
    int64_t leftDim = children[0]->getResultDimensions()[0];
    int64_t rightDim = children[children.size() - 1]->getResultDimensions()[1];
    return {leftDim, rightDim};
  }
  assert(0 && "unreachable");
}

template <typename T>
void Expr::setPropertiesImpl(T *expr) {
  if (expr->isUpperTriangular())
    expr->inferredProperties.insert(Expr::ExprProperty::UPPER_TRIANGULAR);
  if (expr->isLowerTriangular())
    expr->inferredProperties.insert(Expr::ExprProperty::LOWER_TRIANGULAR);
  if (expr->isSquare())
    expr->inferredProperties.insert(Expr::ExprProperty::SQUARE);
  if (expr->isSymmetric())
    expr->inferredProperties.insert(Expr::ExprProperty::SYMMETRIC);
  if (expr->isFullRank())
    expr->inferredProperties.insert(Expr::ExprProperty::FULL_RANK);
  if (expr->isSPD())
    expr->inferredProperties.insert(Expr::ExprProperty::SPD);
}

vector<Expr::ExprProperty> UnaryExpr::getAndSetProperties() {
  vector<Expr::ExprProperty> inferredPropertiesAsVec;
  setPropertiesImpl<UnaryExpr>(this);
  for (auto property : inferredProperties)
    inferredPropertiesAsVec.push_back(property);
  return inferredPropertiesAsVec;
}

vector<Expr::ExprProperty> NaryExpr::getAndSetProperties() {
  vector<Expr::ExprProperty> inferredPropertiesAsVec;
  setPropertiesImpl<NaryExpr>(this);
  for (auto property : inferredProperties)
    inferredPropertiesAsVec.push_back(property);
  return inferredPropertiesAsVec;
}

vector<Expr::ExprProperty> Operand::getProperties() const {
  vector<Expr::ExprProperty> inferredPropertiesAsVec;
  for (auto property : inferredProperties)
    inferredPropertiesAsVec.push_back(property);
  return inferredPropertiesAsVec;
}

ScopedContext *&ScopedContext::getCurrentScopedContext() {
  thread_local ScopedContext *context = nullptr;
  return context;
}

ScopedContext::~ScopedContext() {
  for (auto expr : liveRefs)
    delete expr;
}

/// Check if the shape is square.
bool hasSquareShape(const vector<int64_t> &shape) {
  assert(shape.size() >= 1 && "must be >= 1");
  if (shape.size() == 1)
    return true;
  return all_of(shape.begin(), shape.end(),
                [&](int dim) { return dim == shape[0]; });
}

Operand::Operand(string name, vector<int64_t> shape)
    : ScopedExpr(ExprKind::OPERAND), name(name), shape(shape) {
  if (hasSquareShape(shape))
    this->setProperties({Expr::ExprProperty::SQUARE});
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
    default:
      assert(0 && "UNK");
    }
    if (i != e - 1)
      cout << ", ";
  }
}

/// print the shape of the operand.
static void printShape(vector<int64_t> shape) {
  for (size_t i = 0, e = shape.size(); i < e; i++) {
    cout << shape[i];
    if (i != e - 1)
      cout << ", ";
  }
}

#define LEVEL_SPACES 2

/// Walk a generic expression.
void Expr::walk(int level) const {
  if (auto binaryOp = llvm::dyn_cast_or_null<NaryExpr>(this)) {
    switch (binaryOp->getKind()) {
    case NaryExpr::NaryExprKind::MUL:
      cout << string(level, ' ') << "(*\n";
      break;
    default:
      cout << "UNK";
    }
    for (const Expr *child : binaryOp->getChildren()) {
      child->walk(level + LEVEL_SPACES);
      cout << " \n";
    }
  } // binaryOp
  if (auto unaryOp = llvm::dyn_cast_or_null<UnaryExpr>(this)) {
    switch (unaryOp->getKind()) {
    case UnaryExpr::UnaryExprKind::TRANSPOSE:
      cout << string(level, ' ') << "transpose(";
      break;
    case UnaryExpr::UnaryExprKind::INVERSE:
      cout << string(level, ' ') << "inverse(";
      break;
    default:
      cout << "UNK";
    }
    unaryOp->getChild()->walk(level);
    cout << ")";
  } // unaryOp
  if (auto operand = llvm::dyn_cast_or_null<Operand>(this)) {
    cout << string(level, ' ') << operand->getName() << " [";
    printProperties(operand->getProperties());
    cout << "] [";
    printShape(operand->getShape());
    cout << "]";
  } // operand
}

/// Multiply two or more expressions.
Expr *mlir::linnea::expr::variadicMul(vector<Expr *> children, bool binary) {
  if (binary) {
    assert(children.size() == 2 && "expect only two children");
    return new NaryExpr({children[0], children[1]},
                        NaryExpr::NaryExprKind::MUL);
  }
  // fold other mul inside.
  vector<Expr *> newChildren;
  int size = children.size();
  for (int i = size - 1; i >= 0; i--) {
    if (auto childMul = llvm::dyn_cast_or_null<NaryExpr>(children.at(i))) {
      auto childrenOfChildMul = childMul->getChildren();
      newChildren.insert(newChildren.begin(), childrenOfChildMul.begin(),
                         childrenOfChildMul.end());
    } else
      newChildren.insert(newChildren.begin(), children.at(i));
  }
  return new NaryExpr(newChildren, NaryExpr::NaryExprKind::MUL);
}

/// invert an expression.
Expr *mlir::linnea::expr::inv(Expr *child) {
  assert(child && "child expr must be non null");
  return new UnaryExpr(child, UnaryExpr::UnaryExprKind::INVERSE);
}

/// transpose an expression.
// TODO: check namespace. Why full specify?
Expr *mlir::linnea::expr::trans(Expr *child) {
  assert(child && "child expr must be non null");
  return new UnaryExpr(child, UnaryExpr::UnaryExprKind::TRANSPOSE);
}

static vector<long> getPVector(vector<Expr *> exprs) {
  vector<long> pVector;
  for (auto expr : exprs) {
    Operand *operand = nullptr;
    if (auto unaryOp = llvm::dyn_cast_or_null<UnaryExpr>(expr))
      operand = llvm::dyn_cast_or_null<Operand>(unaryOp->getChild());
    else
      operand = llvm::dyn_cast_or_null<Operand>(expr);
    assert(operand && "must be non null");
    auto shape = operand->getShape();
    if (!pVector.size()) {
      pVector.push_back(shape[0]);
      pVector.push_back(shape[1]);
    } else {
      pVector.push_back(shape[1]);
    }
  }
  return pVector;
}

static void printOptimalParens(const vector<vector<long>> &s, size_t i,
                               size_t j, vector<Expr *> operands) {
  if (i == j) {
    cout << " ";
    Operand *operand = nullptr;
    if (auto unaryOp = llvm::dyn_cast_or_null<UnaryExpr>(operands[i - 1]))
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

static void collectOperandsImpl(Expr *node, vector<Expr *> &operands) {
  if (node) {
    if (auto binaryOp = llvm::dyn_cast_or_null<NaryExpr>(node)) {
      for (auto child : binaryOp->getChildren()) {
        collectOperandsImpl(child, operands);
      }
    }
    if (llvm::isa<UnaryExpr>(node) || llvm::isa<Operand>(node)) {
      assert(node != nullptr && "must be non-null");
      operands.push_back(node);
    }
  }
}

static vector<Expr *> collectOperands(Expr *expr) {
  vector<Expr *> operands;
  collectOperandsImpl(expr, operands);
  return operands;
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

// TODO: n-ary how to handle? Do we need to?
pair<long, long> getKernelCostImpl(Expr *node, long &cost, bool fullTree) {
  if (node) {
    if (auto binaryOp = llvm::dyn_cast_or_null<NaryExpr>(node)) {
      auto children = binaryOp->getChildren();
      // walk(node);
      assert(children.size() == 2 && "expect only two children");
      pair<long, long> left = getKernelCostImpl(children[0], cost, fullTree);
      pair<long, long> right = getKernelCostImpl(children[1], cost, fullTree);
      // note this cost must be the cost of the top level expr
      // not the cost of the tree.
      // GEMM by default adjust later on.
      auto currentCost = left.first * left.second * right.second * 2;
      // TRMM TODO: must be square the other?
      if (children[0]->isLowerTriangular())
        currentCost >>= 1;
      // SYMM TODO: must be square the other?
      else if (children[0]->isSymmetric())
        currentCost >>= 1;

      if (fullTree)
        cost += currentCost;
      else
        cost = currentCost;

      return {left.first, right.second};
    }
    if (auto unaryOp = llvm::dyn_cast_or_null<UnaryExpr>(node)) {
      return getKernelCostImpl(unaryOp->getChild(), cost, fullTree);
    }
    if (auto operand = llvm::dyn_cast_or_null<Operand>(node)) {
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

struct ResultMCP {
  vector<vector<long>> m;
  vector<vector<long>> s;
};

ResultMCP runMCP(Expr *expr) {
#if DEBUG
  cout << "Starting point\n";
  walk(expr);
  cout << "\n\n";
#endif
  vector<Expr *> operands = collectOperands(expr);
  vector<long> pVector = getPVector(operands);
  const size_t n = pVector.size();
  vector<vector<long>> m(n, vector<long>(n, std::numeric_limits<long>::max()));
  vector<vector<long>> s(n, vector<long>(n, std::numeric_limits<long>::max()));

  // store symbolic temporary variables representing sub-chains.
  vector<vector<Expr *>> tmps(n, vector<Expr *>(n, nullptr));

  for (size_t i = 0; i < n - 1; i++)
    tmps[i + 1][i + 1] = operands.at(i);

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

        auto tmpexpr = variadicMul({tmps[i][k], tmps[k + 1][j]}, true);
#if DEBUG
        cout << "---\n";
        walk(tmpexpr);
        cout << "\n---\n\n";
#endif
        long cost = 0;
        getKernelCostTopLevelExpr(tmpexpr, cost);
        q = m[i][k] + m[k + 1][j] + cost;
        if (q < m[i][j]) {
          tmps[i][j] = variadicMul({tmps[i][k], tmps[k + 1][j]}, true);
          // tmps[i][j]->inferProperties();
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
  return {m, s};
}

long Expr::getMCPFlops() {
  ResultMCP result = runMCP(this);
  auto m = result.m;
#if DEBUG
  cout << "FLOPS: " << m[1][m.size() - 1] << "\n";
#endif
  return m[1][m.size() - 1];
}
