
module  {
  func @bar(%arg0: !linnea.matrix<["square"], [30, 15], f32>) -> !linnea.matrix<["square"], [30, 15], f32> {
    %0 = linnea.dummy %arg0 : !linnea.matrix<["square"], [30, 15], f32> -> !linnea.matrix<["square"], [30, 15], f32>
    return %0 : !linnea.matrix<["square"], [30,15], f32>
  }
}

