#include "lstm-compress.h"

#include <algorithm>
#include <math.h>

namespace {
inline float Rand() {
  return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}
inline float Logistic(float val) {
  return 1 / (1 + exp(-val));
}
}

LstmCompress::LstmCompress(unsigned int num_cells, float learning_rate) :
    LstmCompress(num_cells, learning_rate, 0xDEADBEEF) {}

LstmCompress::LstmCompress(unsigned int num_cells, float learning_rate,
    unsigned int seed) : output_(1.0 / 256, 256), state_(num_cells),
    hidden_(num_cells + 1), input_(257 + num_cells), probs_(1.0 / 256, 256),
    hidden_error_(num_cells), tanh_state_(num_cells),
    output_gate_error_(num_cells), output_gate_state_(num_cells),
    state_error_(num_cells), input_node_state_(num_cells),
    input_gate_state_(num_cells), input_node_error_(num_cells),
    input_gate_error_(num_cells), forget_gate_error_(num_cells),
    last_state_(num_cells), forget_gate_state_(num_cells),
    forget_gate_(std::valarray<float>(input_.size()), num_cells),
    input_node_(std::valarray<float>(input_.size()), num_cells),
    input_gate_(std::valarray<float>(input_.size()), num_cells),
    output_gate_(std::valarray<float>(input_.size()), num_cells),
    output_layer_(std::valarray<float>(hidden_.size()), 256),
    learning_rate_(learning_rate) {
  srand(seed);
  hidden_[hidden_.size() - 1] = 1;
  input_[input_.size() - 1] = 1;
  float low = -0.2;
  float range = 0.4;
  for (unsigned int i = 0; i < forget_gate_.size(); ++i) {
    for (unsigned int j = 0; j < forget_gate_[i].size(); ++j) {
      forget_gate_[i][j] = low + Rand() * range;
      input_node_[i][j] = low + Rand() * range;
      input_gate_[i][j] = low + Rand() * range;
      output_gate_[i][j] = low + Rand() * range;
    }
    forget_gate_[i][forget_gate_[i].size() - 1] = 1;
  }
  for (unsigned int i = 0; i < output_layer_.size(); ++i) {
    for (unsigned int j = 0; j < output_layer_[i].size(); ++j) {
      output_layer_[i][j] = low + Rand() * range;
    }
  }
  ForwardPass(0);
}

std::valarray<float>& LstmCompress::Perceive(unsigned char input) {
  BackwardPass(input);
  return Predict(input);
}

std::valarray<float>& LstmCompress::Predict(unsigned char input) {
  ForwardPass(input);
  probs_ = output_;
  double sum = probs_.sum();
  if (sum == 0) {
    probs_ = 1.0 / 256;
  } else {
    probs_ /= sum;
  }
  return probs_;
}

void LstmCompress::ForwardPass(unsigned char input) {
  std::fill_n(begin(input_), 256, 0);
  input_[input] = 1;
  std::copy(begin(hidden_), end(hidden_) - 1, begin(input_) + 256);
  last_state_ = state_;
  for (unsigned int i = 0; i < state_.size(); ++i) {
    forget_gate_state_[i] = Logistic((input_ * forget_gate_[i]).sum());
    state_[i] *= forget_gate_state_[i];
    input_node_state_[i] = Logistic((input_ * input_node_[i]).sum());
    input_gate_state_[i] = tanh((input_ * input_gate_[i]).sum());
    state_[i] += input_node_state_[i] * input_gate_state_[i];
    tanh_state_[i] = tanh(state_[i]);
    output_gate_state_[i] = Logistic((input_ * output_gate_[i]).sum());
    hidden_[i] = output_gate_state_[i] * tanh_state_[i];
  }
  for (unsigned int i = 0; i < 256; ++i) {
    output_[i] = Logistic((hidden_ * output_layer_[i]).sum());
  }
}

void LstmCompress::BackwardPass(unsigned char input) {
  hidden_error_ = 0;
  for (unsigned int i = 0; i < 256; ++i) {
    float error = 0;
    if (i == input) error = (1 - output_[i]);
    else error = -output_[i];
    for (unsigned int j = 0; j < hidden_error_.size(); ++j) {
      hidden_error_[j] += output_layer_[i][j] * error;
    }
    output_layer_[i] += learning_rate_ * error * hidden_;
  }
  output_gate_error_ = tanh_state_ * hidden_error_ * output_gate_state_ *
      (1.0f - output_gate_state_);
  state_error_ += hidden_error_ * output_gate_state_ * (1.0f -
      (tanh_state_ * tanh_state_));
  input_node_error_ = state_error_ * input_gate_state_ * (1.0f -
      (input_node_state_ * input_node_state_));
  input_gate_error_ = state_error_ * input_node_state_ * input_gate_state_ *
      (1.0f - input_gate_state_);
  forget_gate_error_ = state_error_ * last_state_ * forget_gate_state_ *
      (1.0f - forget_gate_state_);
  for (unsigned int i = 0; i < input_node_.size(); ++i) {
    input_node_[i] += (learning_rate_ * input_node_error_[i]) * input_;
    input_gate_[i] += (learning_rate_ * input_gate_error_[i]) * input_;
    forget_gate_[i] += (learning_rate_ * forget_gate_error_[i]) * input_;
    output_gate_[i] += (learning_rate_ * output_gate_error_[i]) * input_;
  }
}
