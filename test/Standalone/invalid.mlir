// RUN: standalone-opt --verify-diagnostics %s 

func @some_func() {
  // expected-error @below {{Unexpected empty region}}
  linnea.equation {

  }
  return 
} 
