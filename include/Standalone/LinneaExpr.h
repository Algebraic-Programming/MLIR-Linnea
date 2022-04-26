//===- LinneaExpr.h --------------------------------------------*- C++ -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MATRIX_CHAIN_UTILS_H
#define MATRIX_CHAIN_UTILS_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallSet.h"
#include <cassert>
#include <memory>
#include <vector>

namespace mlir {
class OpBuilder;
class Location;
class Region;
} // namespace mlir

namespace mlir {
namespace linnea {
namespace expr {

// forward decl.
class Expr;
class NaryExpr;
class UnaryExpr;

/// Scoped context to handle deallocation.
class ScopedContext {
public:
  ScopedContext() { ScopedContext::getCurrentScopedContext() = this; };
  ~ScopedContext();
  ScopedContext(const ScopedContext &) = delete;
  ScopedContext &operator=(const ScopedContext &) = delete;

  void insert(Expr *expr) { liveRefs.insert(expr); }
  void print();
  static ScopedContext *&getCurrentScopedContext();

private:
  llvm::SmallSet<Expr *, 8> liveRefs;
};

/// Generic expr of type BINARY, UNARY, NARY or OPERAND.
class Expr {
public:
  enum class ExprKind { BINARY, UNARY, OPERAND, NARY };
  enum class ExprProperty {
    GENERAL,
    FACTORED,
    UNIT_DIAGONAL,
    DIAGONAL,
    SPSD,
    UPPER_TRIANGULAR,
    LOWER_TRIANGULAR,
    SQUARE,
    SYMMETRIC,
    FULL_RANK,
    SPD
  };

private:
  const ExprKind kind;

protected:
  template <typename T>
  void setPropertiesImpl(T *);
  template <typename T>
  bool hasProperty(T *, Expr::ExprProperty p);
  llvm::SmallSet<Expr::ExprProperty, 8> inferredProperties;

public:
  ExprKind getKind() const { return kind; }
  virtual void setProperties(std::vector<Expr::ExprProperty> properties) {
    assert(0 && "can set properties only for operands");
  };

  virtual ~Expr() = default;

  // properties inference.
  virtual std::vector<Expr::ExprProperty> getAndSetProperties() {
    assert(0 && "cannot call getAndSet properties for operands");
  }

  // return the dimensions of the result. Since we are dealing
  // with only matrices is a two-dimensional vector.
  virtual SmallVector<int64_t, 2> getResultShape() const = 0;

  // query properties.
  virtual bool isUpperTriangular() const = 0;
  virtual bool isLowerTriangular() const = 0;
  virtual bool isSquare() const = 0;
  virtual bool isSymmetric() const = 0;
  virtual bool isFullRank() const = 0;
  virtual bool isSPD() const = 0;
  virtual bool isFactored() const = 0;
  virtual bool isGeneral() const = 0;

  // utilities.
  bool isTransposeOf(const Expr *right);
  bool isSame(const Expr *right);
  void walk(int space = 0) const;
  Expr *simplify(bool symbolicOpt);

protected:
  Expr() = delete;
  Expr(ExprKind kind) : kind(kind), inferredProperties({}){};
};

/// ScopedExpr to automatically manage allocation/deallocation.
template <class T>
class ScopedExpr : public Expr {

private:
  friend T;

public:
  ScopedExpr(Expr::ExprKind kind) : Expr(kind) {
    auto ctx = ScopedContext::getCurrentScopedContext();
    assert(ctx != nullptr && "ctx not available");
    ctx->insert(static_cast<T *>(this));
  }
};

/// Nary operation (i.e., MUL).
// TODO: Does it make sense to query the same
// properties on the "ADD" ?
// TODO: Subclass MUL and ADD similar to Operand and Identity?
class NaryExpr : public ScopedExpr<NaryExpr> {
public:
  enum class NaryExprKind { MUL, ADD };
  enum class SemiringsKind { REAL_ARITH, INTEGER_ARITH, MIN_PLUS, MAX_PLUS };

private:
  std::vector<Expr *> children;
  NaryExprKind kind;
  SemiringsKind semirings;

public:
  NaryExpr() = delete;
  NaryExpr(std::vector<Expr *> children, NaryExprKind kind,
           SemiringsKind semirings)
      : ScopedExpr(ExprKind::BINARY), children(children), kind(kind),
        semirings(semirings){};

  // compute the properties for the current expression.
  std::vector<Expr::ExprProperty> getAndSetProperties() override;

  // get result dimension.
  SmallVector<int64_t, 2> getResultShape() const override;

  // return the children.
  std::vector<Expr *> getChildren() const { return children; }

  // get flops estimate for matrix-chain multiplcation
  // for testing purpose only. Defined only for NaryExprKind::MUL
  long getMCPFlops();

  // query properties.
  bool isUpperTriangular() const override;
  bool isLowerTriangular() const override;
  bool isSquare() const override;
  bool isSymmetric() const override;
  bool isFullRank() const override;
  bool isSPD() const override;
  bool isFactored() const override;
  bool isGeneral() const override;

  // kind of the nary expression.
  NaryExprKind getKind() const { return kind; };

  // get the kind of semirings.
  SemiringsKind getSemiringsKind() const { return semirings; };

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::BINARY;
  };
};

/// Unary operation like transpose or inverse.
class UnaryExpr : public ScopedExpr<UnaryExpr> {
public:
  enum class UnaryExprKind { TRANSPOSE, INVERSE };

private:
  Expr *child;
  UnaryExprKind kind;

public:
  UnaryExpr() = delete;
  UnaryExpr(Expr *child, UnaryExprKind kind)
      : ScopedExpr(ExprKind::UNARY), child(child), kind(kind){};

  // infer properties for the current expression.
  std::vector<Expr::ExprProperty> getAndSetProperties() override;

  // get result dimension.
  SmallVector<int64_t, 2> getResultShape() const override;

  // return the only child.
  Expr *getChild() const { return child; };

  // query properties.
  bool isSquare() const override;
  bool isSymmetric() const override;
  bool isUpperTriangular() const override;
  bool isLowerTriangular() const override;
  bool isFullRank() const override;
  bool isSPD() const override;
  bool isFactored() const override;
  bool isGeneral() const override;

  // kind of unary expression.
  UnaryExprKind getKind() const { return kind; };

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::UNARY;
  };
};

/// Generic operand (i.e., matrix or vector).
class Operand : public ScopedExpr<Operand> {
public:
  enum class OperandKind { MATRIX };

private:
  std::string name;
  SmallVector<int64_t, 2> shape;
  OperandKind kind;

public:
  Operand() = delete;
  Operand(std::string name, SmallVector<int64_t, 2> shape, OperandKind kind);
  std::string getName() const { return name; };
  SmallVector<int64_t, 2> getShape() const { return shape; };
  OperandKind getKind() const { return kind; };

  std::vector<Expr::ExprProperty> getProperties() const;

  // add additional properties to 'inferredProperties'.
  void setProperties(std::vector<Expr::ExprProperty> properties) override;

  // get the dimensionality of the ouput for the current expression.
  SmallVector<int64_t, 2> getResultShape() const override;

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::OPERAND;
  };
};

/// Matrix operand.
class Matrix : public Operand {
public:
  enum class MatrixKind { GENERAL, IDENTITY };

  Matrix() = delete;
  Matrix(std::string name, SmallVector<int64_t, 2> shape,
         MatrixKind kind = MatrixKind::GENERAL);
  MatrixKind getKind() const { return kind; };

  static bool classof(const Expr *expr) {
    if (const Operand *op = llvm::dyn_cast_or_null<Operand>(expr))
      if (op->getKind() == OperandKind::MATRIX)
        return true;
    return false;
  };

  // query properties.
  bool isUpperTriangular() const override;
  bool isLowerTriangular() const override;
  bool isSquare() const override;
  bool isSymmetric() const override;
  bool isFullRank() const override;
  bool isSPD() const override;
  bool isFactored() const override;
  bool isGeneral() const override;

private:
  MatrixKind kind;
};

/// Identity.
class Identity : public Matrix {
public:
  Identity() = delete;
  Identity(SmallVector<int64_t, 2> shape);

  static bool classof(const Expr *expr) {
    if (const Matrix *m = llvm::dyn_cast_or_null<Matrix>(expr))
      if (m->getKind() == MatrixKind::IDENTITY)
        return true;
    return false;
  }

  // query properties.
  bool isUpperTriangular() const override;
  bool isLowerTriangular() const override;
  bool isSquare() const override;
  bool isSymmetric() const override;
  bool isFullRank() const override;
  bool isSPD() const override;
  bool isFactored() const override;
  bool isGeneral() const override;
};

/// Helper class to translate from our internal representation
/// to MLIR IR and vice versa.
class ExprBuilder {
private:
  static thread_local int operandId;

  // map from value to expr.
  llvm::DenseMap<Value, Expr *> valueMap;

  // map from expr to value.
  llvm::DenseMap<Expr *, Value> exprMap;

  // already visited equation operations.
  llvm::DenseSet<Operation *> visited;

  // return the next id for the operand.
  int getNextId() { return operandId++; };

  // build entire expr.
  Expr *buildExprImpl(Value val, Operation *op);

  // build operand as expr.
  Expr *buildOperandImpl(Value val);

  Value buildIRImpl(Location loc, OpBuilder &builder, Expr *root);

  // build linnea.mul/add.low.
  template <typename OP>
  Value buildBinaryOpImpl(Location loc, OpBuilder &builder, NaryExpr *expr);
  // build linnea.transpose/inverse.low.
  template <typename OP>
  Value buildUnaryOpImpl(Location loc, OpBuilder &builder, UnaryExpr *expr);

public:
  // map 'from' to 'to'.
  void map(Value from, Expr *to) { valueMap[from] = to; };
  void map(Expr *from, Value to) { exprMap[from] = to; };

  // check if 'from' is available in valueMap.
  bool contains(Value from) const { return valueMap.count(from); };
  bool contains(Expr *from) const { return exprMap.count(from); };
  bool isAlreadyVisited(Operation *op) { return visited.count(op); };

  // return value given 'from' key (must be available).
  Expr *lookup(Value from) {
    assert(contains(from) && "expect value to be contained in the map");
    return valueMap[from];
  }
  Value lookup(Expr *from) {
    assert(contains(from) && "expect value to be contained in the map");
    return exprMap[from];
  }

  // build a linnea symbolic expression from a Value
  // by walking use-def chain.
  Expr *buildLinneaExpr(Value value, Operation *op);

  // build mlir IR from a linnea symbolic expression
  // by walking the expression.
  Value buildIR(Location loc, OpBuilder &builder, Expr *root);

  ExprBuilder() = default;
  // FIXME: remove this. Needed beacuse we call root->simplify()
  ExprBuilder(bool runSymbolicOpt) : symbolicOpt(runSymbolicOpt){};
  ExprBuilder(ExprBuilder &) = delete;

private:
  bool symbolicOpt = false;
};

template <typename K>
Expr *variadicExpr(std::vector<Expr *> children, bool isBinary);
template <typename K>
Expr *unaryExpr(Expr *child);

} // end namespace expr.
} // end namespace linnea.
} // end namespace mlir.

#include "Standalone/LinneaExprImpl.h"

#endif
