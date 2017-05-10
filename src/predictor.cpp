#include "predictor.h"
#include "lstm/byte-model.h"

Predictor::Predictor() : lstm_(40, 3, 10, 0.1) {
  srand(0xDEADBEEF);
}

float Predictor::Predict() {
  return lstm_.Predict();
}

void Predictor::Perceive(int bit) {
  lstm_.Perceive(bit);
}
