# TODO
- linnea fillOp using assembly format (again)
- https://docs.oracle.com/cd/E19957-01/806-3566/plug_matrices.html
- http://blog.ezyang.com/2019/05/pytorch-internals/
- https://ezyang.github.io/stride-visualizer/index.html
- https://documentation.suse.com/sle-rt/15-SP2/html/SLE-RT-all/cha-shielding-cpuset.html

- API
numpy.tril() API
numpy.triu() API
numpy.diag() API
numpy.identity() API

# An out-of-tree dialect template for MLIR

This repository contains a template for an out-of-tree [MLIR](https://mlir.llvm.org/) dialect as well as a
standalone `opt`-like tool to operate on that dialect.

## How to build

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-standalone-opt
```
To build the documentation from the TableGen description of the dialect
operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with
CMake so that it installs `FileCheck` to the chosen installation prefix.

## License

This dialect template is made available under the Apache License 2.0 with LLVM Exceptions. See the `LICENSE.txt` file for more details.

## 27th of October

Example:
```mlir
  %symb = linnea.equation %symbol1 : matrix<[2,2]> {
            ^bb0(%a : matrix<[2,2]):
              %do something
              (body only symbolic)
              linnea.yield %something
              (nice to have linnea.equal instead of linnea.yield)
          }
  use %symbol

Type: AnySymbol, Matrix[size][properties], vector[size][properties], scalar.

What should be the type of the returned value? AnyType? or AnySymbol?

```

## 28th of Sept
Current idea is to drop the tensor and attched the dimension information on the attribute.

```
#S = #linnea.matrix_encoding<{p = ["s"]}, {d = [32, 32]}, {id: S}>
#A = #linnea.matrix_encoding<{p = ["ge"]}, {d = [32, 32]}, {id: A}>

func private @check_type(%arg0: !linnea.matrix<#S>, %arg1: !linnea.matrix<#A>) {
	%0 = transpose(%arg0) : !linnea.matrix<#A> // need to create a new type AT if "ge" changes (i.e., transpose of a lowerT)
	%1 = mul(%0, %arg1, %arg0) : !linnea.matrix<#AT>, !linnea.matrix<#S>, !linnea.matrix<#A>
	return %1
}
```

## 30th of Sept

- Rule propagation scale?
- Choice of rule to fire?

- Assumption "f" factorized flag does not get propagate by a mul.

```
C = A^T * S^(-1) * A
C = A^(T) * (L L^T)^(-1) * A
C = A^(T) * L^(-T) * L^(-1) * A

func(%A : matrix<["frank, sq"]>, %S : matrix<["spd"]>) {
  %0 = trans(%A) : matrix<["frank, sq"]> -> matrix<["frank,sq"]>
  %1 = inv(%S) : matrix<["spd"]> -> matrix<["spd"]>
  %2 = mul(%0, %1, %A) : matrix<["frank, sq"]>, matrix<["spd"]>, matrix<["frank, sq"]> -> matrix<["spd"]>
}

Rule 1. factorization: operand mul comes from inverse

func(%A : matrix<["frank, sq"]>, %S : matrix<["spd"]>) {
  %0 = trans(%A) : matrix<["frank, sq"]> -> matrix<["frank, sq"]>
	
  %1 = chol(%S) : matrix<["spd"]> -> matrix<["lt", "f"]>
  %2 = trans(%1) : matrix<["lt", "f"]> -> matrix<["ut", "f"]>
  %3 = mul(%1, %2) : matrix<["lt", "f"]>, matrix<["ut", "f"]> -> matrix<["spd", "f"]>
  %4 = inv(%3) : matrix<["spd", "f"]> -> matrix<["spd", "f"]>
  
  %5 = mul(%0, %4, %A) : matrix<["frank, sq"]>, matrix<["spd"]>, matrix<["frank, sq"]> -> matrix<["spd"]>
}

Rule 2. inverse of a product

func(%A : matrix<["frank, sq"]>, %S : matrix<["spd"]>) {
  %0 = trans(%A) : matrix<["frank, sq"]> -> matrix<["frank, sq"]>
	
  %1 = chol(%S) : matrix<["spd"]> -> matrix<["lt", "f"]>
  %2 = trans(%1) : matrix<["lt", "f"]> -> matrix<["ut", "f"]>
  
  %3 = inv(%2) : matrix<["ut", "f"]> -> matrix<["lt", "f"]>
  %4 = inv(%1) : matrix<["lt", "f"]> -> matrix<["ut", "f"]>
  %5 = mul(%3, %4) : matrix<["lt", "f"]>, matrix<["ut", "f"]> -> matrix<["spd"]> 
  
  %6 = mul(%0, %5, %A) : matrix<["frank, sq"]>, matrix<["spd"]>, matrix<["frank, sq"]> -> matrix<["spd"]>
}

Rule 3. collapse mul together

func(%A : matrix<["frank, sq"], %S : matrix<["spd"]) {
  %0 = trans(%A) : matrix<["frank, sq"]> -> matrix<["frank, sq"]>
	
  %1 = chol(%S) : matrix<["spd"]> -> matrix<["lt", "f"]>
  %2 = trans(%1) : matrix<["lt", "f"]> -> matrix<["ut", "f"]>
  
  %3 = inv(%2) : matrix<["ut", "f"]> -> matrix<["lt", "f"]>
  %4 = inv(%1) : matrix<["lt", "f"]> -> matrix<["ut", "f"]>
  %5 = mul(%0, %3, %4, %A) : matrix<["frank, sq"]>, matrix<["lt", "f"]>, matrix<["ut", "f"]>,
  			                     matrix<["frank, sq"]> -> matrix<["spd"]>
}
```
