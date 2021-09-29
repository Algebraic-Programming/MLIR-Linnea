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
