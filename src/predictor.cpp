#include "predictor.h"
#include "lstm/byte-model.h"

Predictor::Predictor() : lstm_(80, 3, 20, 0.05) {
  srand(0xDEADBEEF);
}

float Predictor::Predict() {
  return lstm_.Predict();
}

void Predictor::Perceive(int bit) {
  lstm_.Perceive(bit);
}
