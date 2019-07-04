#include "predictor.h"
#include "lstm/byte-model.h"
#include "lstm/lstm.h"

Predictor::Predictor(const std::vector<bool>& vocab) : vocab_(vocab) {
  srand(0xDEADBEEF);
  unsigned int vocab_size = 0;
  for (unsigned int i = 0; i < vocab_.size(); ++i) {
    if (vocab_[i]) ++vocab_size;
  }
  lstm_.reset(new ByteModel(vocab_,
      new Lstm(0, vocab_size, 90, 3, 10, 0.05, 2)));
}

float Predictor::Predict() {
  return lstm_->Predict();
}

void Predictor::Perceive(int bit) {
  lstm_->Perceive(bit);
}
