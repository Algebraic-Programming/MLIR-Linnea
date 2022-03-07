#include "basicOp.h"
#include "gtest/gtest.h"

using namespace std;
using namespace mlir::linnea::expr;

TEST(Basic, mul) { EXPECT_EQ(0, 0); }

TEST(Basic, conversion) {
  ScopedContext ctx;
  Operand *a = new Matrix("A1", {30, 35});
  EXPECT_TRUE(llvm::isa<Matrix>(a));
  Operand *i = new Identity({30, 30});
  EXPECT_TRUE(llvm::isa<Identity>(i));
  EXPECT_FALSE(llvm::isa<Identity>(a));
  EXPECT_TRUE(llvm::isa<Matrix>(i));
}
