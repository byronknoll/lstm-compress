#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "lstm/byte-model.h"

class Predictor {
 public:
  Predictor();
  float Predict();
  void Perceive(int bit);

 private:
  ByteModel lstm_;
};

#endif
