#include "lstm-compress.h"

namespace {
inline float Rand() {
  return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}
inline float Logistic(float val) {
  return 1 / (1 + exp(-val));
}
}

LstmCompress::LstmCompress(unsigned int num_cells, unsigned int num_layers,
    float learning_rate) : output_(1.0 / 256, 256), probs_(1.0 / 256, 256),
    hidden_(num_cells * num_layers + 1), hidden_error_(num_cells),
    layer_input_(std::valarray<float>(257 + num_cells * 2), num_layers),
    output_layer_(std::valarray<float>(num_cells * num_layers + 1), 256),
    learning_rate_(learning_rate), num_cells_(num_cells) {
  hidden_[hidden_.size() - 1] = 1;
  layer_input_[0].resize(257 + num_cells);
  for (unsigned int i = 0; i < num_layers; ++i) {
    layer_input_[i][layer_input_[i].size() - 1] = 1;
    layers_.push_back(std::unique_ptr<Layer>(new Layer(layer_input_[i].size(),
        num_cells, learning_rate)));
  }
  float low = -0.2;
  float range = 0.4;
  for (unsigned int i = 0; i < output_layer_.size(); ++i) {
    for (unsigned int j = 0; j < output_layer_[i].size(); ++j) {
      output_layer_[i][j] = low + Rand() * range;
    }
  }
  Predict(0);
}

std::valarray<float>& LstmCompress::Perceive(unsigned char input) {
  for (int layer = layers_.size() - 1; layer >= 0; --layer) {
    int offset = layer * num_cells_;
    for (unsigned int i = 0; i < 256; ++i) {
      float error = 0;
      if (i == input) error = (1 - output_[i]);
      else error = -output_[i];
      for (unsigned int j = 0; j < hidden_error_.size(); ++j) {
        hidden_error_[j] += output_layer_[i][j + offset] * error;
      }
    }
    hidden_error_ = layers_[layer]->BackwardPass(layer_input_[layer],
        hidden_error_);
  }
  for (unsigned int i = 0; i < 256; ++i) {
    float error = 0;
    if (i == input) error = (1 - output_[i]);
    else error = -output_[i];
    output_layer_[i] += learning_rate_ * error * hidden_;
  }
  return Predict(input);
}

std::valarray<float>& LstmCompress::Predict(unsigned char input) {
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    std::fill_n(begin(layer_input_[i]), 256, 0);
    layer_input_[i][input] = 1;
    auto start = begin(hidden_) + i * num_cells_;
    std::copy(start, start + num_cells_, begin(layer_input_[i]) + 256);
    const auto& hidden = layers_[i]->ForwardPass(layer_input_[i]);
    std::copy(begin(hidden), end(hidden), start);
    if (i < layers_.size() - 1) {
      start = begin(layer_input_[i + 1]) + 256 + num_cells_;
      std::copy(begin(hidden), end(hidden), start);
    }
  }
  for (unsigned int i = 0; i < 256; ++i) {
    output_[i] = Logistic((hidden_ * output_layer_[i]).sum());
  }
  probs_ = output_;
  double sum = 0, min = 0.000001;
  for (int i = 0; i < 256; ++i) {
    if (probs_[i] < min) probs_[i] = min;
    sum += probs_[i];
  }
  probs_ /= sum;
  return probs_;
}
