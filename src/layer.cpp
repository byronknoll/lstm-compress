#include "layer.h"

#include <math.h>
#include <algorithm>

namespace {
inline float Rand() {
  return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}
inline float Logistic(float val) {
  return 1 / (1 + exp(-val));
}
}

Layer::Layer(unsigned int num_input, unsigned int num_cells,
    float learning_rate) : state_(num_cells), hidden_(num_cells),
    hidden_error_(num_cells), tanh_state_(num_cells),
    output_gate_error_(num_cells), output_gate_state_(num_cells),
    state_error_(num_cells), input_node_state_(num_cells),
    input_gate_state_(num_cells), input_node_error_(num_cells),
    input_gate_error_(num_cells), forget_gate_error_(num_cells),
    last_state_(num_cells), forget_gate_state_(num_cells),
    forget_gate_(std::valarray<float>(num_input), num_cells),
    input_node_(std::valarray<float>(num_input), num_cells),
    input_gate_(std::valarray<float>(num_input), num_cells),
    output_gate_(std::valarray<float>(num_input), num_cells),
    learning_rate_(learning_rate), num_cells_(num_cells) {
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
}

const std::valarray<float>& Layer::ForwardPass(const std::valarray<float>&
    input) {
  last_state_ = state_;
  for (unsigned int i = 0; i < state_.size(); ++i) {
    forget_gate_state_[i] = Logistic((input * forget_gate_[i]).sum());
    state_[i] *= forget_gate_state_[i];
    input_node_state_[i] = tanh((input * input_node_[i]).sum());
    input_gate_state_[i] = Logistic((input * input_gate_[i]).sum());
    state_[i] += input_node_state_[i] * input_gate_state_[i];
    tanh_state_[i] = tanh(state_[i]);
    output_gate_state_[i] = Logistic((input * output_gate_[i]).sum());
    hidden_[i] = output_gate_state_[i] * tanh_state_[i];
  }
  return hidden_;
}

const std::valarray<float>& Layer::BackwardPass(const std::valarray<float>&
    input, const std::valarray<float>& hidden_error) {
  output_gate_error_ = tanh_state_ * hidden_error * output_gate_state_ *
      (1.0f - output_gate_state_);
  state_error_ = hidden_error * output_gate_state_ * (1.0f -
      (tanh_state_ * tanh_state_));
  input_node_error_ = state_error_ * input_gate_state_ * (1.0f -
      (input_node_state_ * input_node_state_));
  input_gate_error_ = state_error_ * input_node_state_ * input_gate_state_ *
      (1.0f - input_gate_state_);
  forget_gate_error_ = state_error_ * last_state_ * forget_gate_state_ *
      (1.0f - forget_gate_state_);

  hidden_error_ = 0;
  if (input.size() > 257 + num_cells_) {
    int offset = 256 + num_cells_;
    for (unsigned int i = 0; i < input_node_.size(); ++i) {
      for (unsigned int j = offset; j < input.size() - 1; ++j) {
        hidden_error_[j-offset] += input_node_[i][j] * input_node_error_[i];
        hidden_error_[j-offset] += input_gate_[i][j] * input_gate_error_[i];
        hidden_error_[j-offset] += forget_gate_[i][j] * forget_gate_error_[i];
        hidden_error_[j-offset] += output_gate_[i][j] * output_gate_error_[i];
      }
    }
  }

  for (unsigned int i = 0; i < input_node_.size(); ++i) {
    input_node_[i] += (learning_rate_ * input_node_error_[i]) * input;
    input_gate_[i] += (learning_rate_ * input_gate_error_[i]) * input;
    forget_gate_[i] += (learning_rate_ * forget_gate_error_[i]) * input;
    output_gate_[i] += (learning_rate_ * output_gate_error_[i]) * input;
  }
  return hidden_error_;
}
