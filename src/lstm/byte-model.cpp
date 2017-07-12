#include "byte-model.h"
#include "lstm.h"

ByteModel::ByteModel(unsigned int num_cells, unsigned int num_layers,
    int horizon, float learning_rate) : top_(255), mid_(0), bot_(0),
    probs_(1.0 / 256, 256), bit_context_(1), lstm_(0, 256, num_cells,
    num_layers, horizon, learning_rate) {}

float ByteModel::Predict() {
  float num = 0, denom = 0;
  mid_ = bot_ + ((top_ - bot_) / 2);
  for (int i = bot_; i <= top_; ++i) {
    denom += probs_[i];
    if (i > mid_) num += probs_[i];
  }
  if (denom == 0) return 0.5;
  return num / denom;
}

void ByteModel::Perceive(int bit) {
  if (bit) {
    bot_ = mid_ + 1;
  } else {
    top_ = mid_;
  }
  bit_context_ += bit_context_ + bit;
  if (bit_context_ >= 256) {
    bit_context_ -= 256;
    probs_ = lstm_.Perceive(bit_context_);
    bit_context_ = 1;
    top_ = 255;
    bot_ = 0;
  }
}

