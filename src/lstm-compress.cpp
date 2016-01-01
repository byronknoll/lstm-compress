#include "lstm-compress.h"

#include <algorithm>
#include <math.h>

LstmCompress::LstmCompress(unsigned int num_cells, float learning_rate) :
    output_(1.0 / 256, 256), state_(num_cells), hidden_(num_cells + 1),
    input_(257 + num_cells), probs_(1.0 / 256, 256),
    forget_gate_(std::valarray<float>(0.0, input_.size()), num_cells),
    input_gate_(std::valarray<float>(0.0, input_.size()), num_cells),
    candidate_gate_(std::valarray<float>(0.0, input_.size()), num_cells),
    output_gate_(std::valarray<float>(0.0, input_.size()), num_cells),
    output_layer_(std::valarray<float>(0.0, hidden_.size()), 256),
    learning_rate_(learning_rate) {
  hidden_[hidden_.size() - 1] = 1;
  input_[input_.size() - 1] = 1;
}

std::valarray<float>& LstmCompress::Perceive(unsigned char input) {
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

inline float Logistic(float val) {
  return 1 / (1 + exp(-val));
}

void LstmCompress::ForwardPass(unsigned char input) {
  std::fill_n(begin(input_), 256, 0);
  input_[input] = 1;
  std::copy(begin(hidden_), end(hidden_), begin(input_) + 256);
  for (unsigned int i = 0; i < state_.size(); ++i) {
    state_[i] *= Logistic((input_ * forget_gate_[i]).sum());
    state_[i] += Logistic((input_ * input_gate_[i]).sum()) *
        (4 * Logistic((input_ * candidate_gate_[i]).sum()) - 2);
    hidden_[i] = (Logistic((input_ * output_gate_[i]).sum())) *
        (2 * Logistic(state_[i]) - 1);
  }
  for (unsigned int i = 0; i < 256; ++i) {
    output_[i] = Logistic((hidden_ * output_layer_[i]).sum());
  }
}

