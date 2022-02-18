//===- StandaloneExtension.cpp - Extension module -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone-c/LinneaDialect.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_standaloneDialects, m) {
  //===--------------------------------------------------------------------===//
  // standalone dialect
  //===--------------------------------------------------------------------===//
  auto standalone_m = m.def_submodule("standalone");

  standalone_m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__linnea__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  // Linnea MatrixType.
  auto MatrixType =
      mlir_type_subclass(m, "MatrixType", mlirTypeIsLinneaMatrixType);
  MatrixType.def_classmethod(
      "get",
      [](py::object cls, MlirContext ctx, MlirAttribute attr,
         std::vector<int64_t> shape, MlirType elementType) {
        return cls(mlirLinneaMatrixTypeGet(ctx, attr, shape.size(),
                                           shape.data(), elementType));
      },
      py::arg("cls"), py::arg("ctx"), py::arg("attr"), py::arg("shape"),
      py::arg("elementType"), "Create a MatrixType");

  // Linnea MatrixEncodingAttr.
  py::enum_<MlirLinneaMatrixEncoding>(m, "Property", py::module_local())
      .value("general", MLIR_LINNEA_MATRIX_PROPERTY_GENERAL)
      .value("fullrank", MLIR_LINNEA_MATRIX_PROPERTY_FULLRANK)
      .value("diagonal", MLIR_LINNEA_MATRIX_PROPERTY_DIAGONAL)
      .value("unitdiagonal", MLIR_LINNEA_MATRIX_PROPERTY_UNITDIAGONAL)
      .value("lowertriangular", MLIR_LINNEA_MATRIX_PROPERTY_LOWERTRIANGULAR)
      .value("uppertriangular", MLIR_LINNEA_MATRIX_PROPERTY_UPPERTRIANGULAR)
      .value("symmetric", MLIR_LINNEA_MATRIX_PROPERTY_SYMMETRIC)
      .value("spd", MLIR_LINNEA_MATRIX_PROPERTY_SPD)
      .value("spsd", MLIR_LINNEA_MATRIX_PROPERTY_SPSD)
      .value("square", MLIR_LINNEA_MATRIX_PROPERTY_SQUARE)
      .value("factored", MLIR_LINNEA_MATRIX_PROPERTY_FACTORED);

  auto MatrixEncodingAttr = mlir_attribute_subclass(
      m, "LinneaMatrixEncodingAttr", mlirAttributeIsLinneaMatrixEncodingAttr);

  MatrixEncodingAttr.def_classmethod(
      "get",
      [](py::object cls, MlirContext ctx,
         std::vector<MlirLinneaMatrixEncoding> properties) {
        return cls(mlirLinneaAttributeMatrixEncodingAttrGet(
            ctx, properties.size(), properties.data()));
      },
      py::arg("cls"), py::arg("ctx"), py::arg("properties"),
      "Gets a matrix type encoding.");
}
