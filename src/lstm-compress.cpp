#include "lstm-compress.h"

#include <algorithm>
#include <math.h>

LstmCompress::LstmCompress(unsigned int num_cells) : output_(256),
    state_(num_cells), hidden_(num_cells + 1), input_(257 + num_cells) {
  output_ = 1.0 / 256;
  state_ = 0;
  hidden_ = 0;
  hidden_[hidden_.size() - 1] = 1;
  input_ = 0;
  input_[input_.size() - 1] = 1;
  forget_gate_.resize(num_cells);
  input_gate_.resize(num_cells);
  candidate_gate_.resize(num_cells);
  output_gate_.resize(num_cells);
  for (unsigned int i = 0; i < num_cells; ++i) {
    forget_gate_[i].resize(input_.size());
    input_gate_[i].resize(input_.size());
    candidate_gate_[i].resize(input_.size());
    output_gate_[i].resize(input_.size());
  }
  output_layer_.resize(256);
  for (unsigned int i = 0; i < 256; ++i) {
    output_layer_[i].resize(hidden_.size());
  }
}

std::valarray<float>& LstmCompress::Perceive(unsigned char val) {
  return output_;
}

void LstmCompress::ForwardPass(unsigned char input) {
  std::fill_n(begin(input_), 256, 0);
  input_[input] = 1;
  std::copy(begin(hidden_), end(hidden_), begin(input_) + 256);
  for (unsigned int i = 0; i < state_.size(); ++i) {
    state_[i] *= 1 / (1 + exp(-(input_ * forget_gate_[i]).sum()));
    state_[i] += (1 / (1 + exp(-(input_ * input_gate_[i]).sum()))) *
        (tanh((input_ * candidate_gate_[i]).sum()));
    hidden_[i] = (1 / (1 + exp(-(input_ * output_gate_[i]).sum()))) *
        tanh(state_[i]);
  }
  for (unsigned int i = 0; i < 256; ++i) {
    output_[i] = (1 / (1 + exp(-(hidden_ * output_layer_[i]).sum())));
  }
}

