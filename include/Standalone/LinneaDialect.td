//===- LinneaDialect.td - Linnea dialect -------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINNEA_DIALECT
#define LINNEA_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Linnea dialect definition.
//===----------------------------------------------------------------------===//

def Linnea_Dialect : Dialect {
    let name = "linnea";
    let summary = "A linnea out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::linnea";

     let extraClassDeclaration = [{
        //Type parseType(DialectAsmParser& parser) const override;
        //void printType(Type type, DialectAsmPrinter& printer) const override;
        Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;
        void printAttribute(Attribute attr, DialectAsmPrinter &printer) const override;
    }];
}

//===----------------------------------------------------------------------===//
// Base linnea operation definition.
//===----------------------------------------------------------------------===//

class Linnea_Op<string mnemonic, list<Trait> traits = []> :
        Op<Linnea_Dialect, mnemonic, traits>;

#endif // LINNEA_DIALECT
