#include "byte-model.h"
#include "lstm.h"

#include <numeric>

ByteModel::ByteModel(const std::vector<bool>& vocab, Lstm* lstm) : top_(255),
    mid_(0), bot_(0), byte_map_(0, 256), probs_(1.0 / 256, 256),
    bit_context_(1), lstm_(lstm), vocab_(vocab) {
  int offset = 0;
  for (int i = 0; i < 256; ++i) {
    byte_map_[i] = offset;
    if (vocab_[i]) ++offset;
  }
}

float ByteModel::Predict() {
  mid_ = bot_ + ((top_ - bot_) / 2);
  float num = std::accumulate(&probs_[mid_ + 1], &probs_[top_ + 1], 0.0f);
  float denom = std::accumulate(&probs_[bot_], &probs_[mid_ + 1], num);
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
    const auto& output = lstm_->Perceive(byte_map_[bit_context_]);
    int offset = 0;
    for (int i = 0; i < 256; ++i) {
      if (vocab_[i]) {
        probs_[i] = output[offset];
        ++offset;
      } else {
        probs_[i] = 0;
      }
    }
    bit_context_ = 1;
    top_ = 255;
    bot_ = 0;
  }
}

