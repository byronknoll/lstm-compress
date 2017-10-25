#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "lstm/byte-model.h"

class Predictor {
 public:
  Predictor(const std::vector<bool>& vocab);
  float Predict();
  void Perceive(int bit);

 private:
  std::unique_ptr<ByteModel> lstm_;
  std::vector<bool> vocab_;
};

#endif
