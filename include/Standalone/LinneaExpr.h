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

/// Generic expr of type BINARY, UNARY or OPERAND.
class Expr {
public:
  enum class ExprKind { BINARY, UNARY, OPERAND, NARY };
  enum class ExprProperty {
    GENERAL,
    FACTORED,
    UNIT_DIAGONAL,
    DIAGONAL,
    SPSD,
    IDENTITY,
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

  // normal form.
  virtual Expr *getNormalForm() = 0;

  // get result dimension.
  virtual std::vector<int64_t> getResultDimensions() const = 0;

  // properties.
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
  long getMCPFlops();
  Expr *simplify();

protected:
  Expr() = delete;
  Expr(ExprKind kind) : kind(kind), inferredProperties({}){};
};

/// ScopedExpr
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
class NaryExpr : public ScopedExpr<NaryExpr> {
public:
  enum class NaryExprKind { MUL, ADD };

private:
  std::vector<Expr *> children;
  NaryExprKind kind;

public:
  NaryExpr() = delete;
  NaryExpr(std::vector<Expr *> children, NaryExprKind kind)
      : ScopedExpr(ExprKind::BINARY), children(children), kind(kind){};

  // compute the properties for the current expression.
  std::vector<Expr::ExprProperty> getAndSetProperties() override;

  // get the normal form for the current expression.
  Expr *getNormalForm() override;

  // get result dimension.
  std::vector<int64_t> getResultDimensions() const override;

  // return the children.
  std::vector<Expr *> getChildren() const { return children; }

  // properties.
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

  // get the normal form for the current expression.
  Expr *getNormalForm() override;

  // get result dimension.
  std::vector<int64_t> getResultDimensions() const override;

  // return the only child.
  Expr *getChild() const { return child; };

  // properties.
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
private:
  std::string name;
  std::vector<int64_t> shape;

public:
  Operand() = delete;
  Operand(std::string name, std::vector<int64_t> shape);
  std::string getName() const { return name; };
  std::vector<int64_t> getShape() const { return shape; };

  std::vector<Expr::ExprProperty> getProperties() const;

  // add additional properties to 'inferredProperties'.
  void setProperties(std::vector<Expr::ExprProperty> properties) override;

  // return normal form for current expression.
  Expr *getNormalForm() override;

  // get the dimensionality of the ouput for the current expression.
  std::vector<int64_t> getResultDimensions() const override;

  // properties
  bool isUpperTriangular() const override;
  bool isLowerTriangular() const override;
  bool isSquare() const override;
  bool isSymmetric() const override;
  bool isFullRank() const override;
  bool isSPD() const override;
  bool isFactored() const override;
  bool isGeneral() const override;

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::OPERAND;
  };
};

class ExprBuilder {
private:
  static thread_local int operandId;

  // map from value to expr.
  llvm::DenseMap<Value, Expr *> valueMap;

  // map from expr to value.
  llvm::DenseMap<Expr *, Value> exprMap;

  // return the next id for the operand.
  int getNextId() { return operandId++; };

  // build entire expr.
  Expr *buildExprImpl(mlir::Value val);

  // build operand as expr.
  Expr *buildOperandImpl(mlir::Value type);

  // build mul/transpose/inverse.
  mlir::Value buildIRImpl(Location loc, OpBuilder &builder, Expr *root);
  mlir::Value buildMulImpl(Location loc, OpBuilder &builder, NaryExpr *expr);
  mlir::Value buildTransposeImpl(Location loc, OpBuilder &builder,
                                 UnaryExpr *expr);
  mlir::Value buildInverseImpl(Location loc, OpBuilder &builder,
                               UnaryExpr *expr);

  // map 'from' to 'to'.
  void map(Value from, Expr *to) { valueMap[from] = to; };
  void map(Expr *from, Value to) { exprMap[from] = to; };

  // check if 'from' is available in valueMap.
  bool contains(Value from) const { return valueMap.count(from); };
  bool contains(Expr *from) const { return exprMap.count(from); };

  // return value given 'from' key (must be available).
  Expr *lookup(Value from) {
    assert(contains(from) && "expect value to be contained in the map");
    return valueMap[from];
  }
  Value lookup(Expr *from) {
    assert(contains(from) && "expect value to be contained in the map");
    return exprMap[from];
  }

public:
  Expr *buildLinneaExpr(mlir::Value value);
  mlir::Value buildIR(Location loc, OpBuilder &builder, Expr *root);

  ExprBuilder() = default;
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
