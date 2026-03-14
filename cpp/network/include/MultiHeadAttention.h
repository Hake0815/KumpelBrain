#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include "../include/SaveLoadMixin.h"
#include <torch/torch.h>

struct MultiHeadAttentionImpl : torch::nn::Module,
                                SaveLoadMixin<MultiHeadAttentionImpl> {
  MultiHeadAttentionImpl(int64_t d_q, int64_t d_k, int64_t d_v, int64_t d_head,
                         int64_t nheads, double dropout = 0.0, bool bias = true,
                         torch::Device device = torch::kCPU,
                         torch::Dtype dtype = torch::kFloat);

  torch::Tensor
  forward(const torch::Tensor &query, const torch::Tensor &key,
          const torch::Tensor &value,
          torch::optional<torch::Tensor> attn_mask = torch::nullopt,
          bool is_causal = false);

private:
  int64_t d_head_;
  int64_t nheads_;
  double dropout_p_;
  bool qkv_same_embed_dim_;
  bool use_bias_;

  torch::nn::Linear packed_proj_{nullptr};
  torch::nn::Linear q_proj_{nullptr};
  torch::nn::Linear k_proj_{nullptr};
  torch::nn::Linear v_proj_{nullptr};
  torch::nn::Linear out_proj_{nullptr};
  torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(MultiHeadAttention);

#endif