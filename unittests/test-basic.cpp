#include "basicOp.h"
#include "gtest/gtest.h"

using namespace std;
using namespace mlir::linnea::expr;

TEST(Basic, mul) { EXPECT_EQ(0, 0); }

TEST(Basic, conversion) {
  ScopedContext ctx;
  Operand *A = new Matrix("A1", {30, 35});
  EXPECT_TRUE(llvm::isa<Matrix>(A));
  Operand *I = new Identity({30, 30});
  EXPECT_TRUE(llvm::isa<Identity>(I));
  EXPECT_FALSE(llvm::isa<Identity>(A));
  EXPECT_TRUE(llvm::isa<Matrix>(I));
}
