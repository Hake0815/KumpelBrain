#include "../include/MultiHeadAttention.h"
#include <torch/nn/functional.h>
#include <torch/serialize.h>
#include <torch/torch.h>

MultiHeadAttentionImpl::MultiHeadAttentionImpl(int64_t d_q, int64_t d_k,
                                               int64_t d_v, int64_t d_head,
                                               int64_t nheads, double dropout,
                                               bool bias, torch::Device device,
                                               torch::Dtype dtype) {
  d_head_ = d_head;
  nheads_ = nheads;
  dropout_p_ = dropout;
  use_bias_ = bias;
  qkv_same_embed_dim_ = (d_q == d_k && d_q == d_v);

  const int64_t d_total = d_head * nheads;
  const int64_t d_out = d_q;

  if (qkv_same_embed_dim_) {
    packed_proj_ = register_module(
        "packed_proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(d_q, d_total * 3).bias(bias)));
  } else {
    q_proj_ = register_module(
        "q_proj",
        torch::nn::Linear(torch::nn::LinearOptions(d_q, d_total).bias(bias)));
    k_proj_ = register_module(
        "k_proj",
        torch::nn::Linear(torch::nn::LinearOptions(d_k, d_total).bias(bias)));
    v_proj_ = register_module(
        "v_proj",
        torch::nn::Linear(torch::nn::LinearOptions(d_v, d_total).bias(bias)));
  }

  out_proj_ = register_module(
      "out_proj",
      torch::nn::Linear(torch::nn::LinearOptions(d_total, d_out).bias(bias)));

  dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
  to(device, dtype);
}

torch::Tensor MultiHeadAttentionImpl::forward(
    const torch::Tensor &query, const torch::Tensor &key,
    const torch::Tensor &value, torch::optional<torch::Tensor> attn_mask,
    bool is_causal) {
  torch::Tensor query_projected, key_projected, value_projected;

  // Step 1. Apply input projection
  if (qkv_same_embed_dim_) {
    if (query.data_ptr() == key.data_ptr() &&
        key.data_ptr() == value.data_ptr()) {
      auto result = packed_proj_(query);
      auto chunks = result.chunk(3, -1);
      query_projected = chunks[0];
      key_projected = chunks[1];
      value_projected = chunks[2];
    } else {
      auto weight = packed_proj_->weight;
      auto bias = packed_proj_->bias;
      auto w_chunks = weight.chunk(3, 0);
      if (use_bias_) {
        auto bias_chunks = bias.chunk(3, 0);
        query_projected =
            torch::nn::functional::linear(query, w_chunks[0], bias_chunks[0]);
        key_projected =
            torch::nn::functional::linear(key, w_chunks[1], bias_chunks[1]);
        value_projected =
            torch::nn::functional::linear(value, w_chunks[2], bias_chunks[2]);
      } else {
        query_projected = torch::nn::functional::linear(query, w_chunks[0]);
        key_projected = torch::nn::functional::linear(key, w_chunks[1]);
        value_projected = torch::nn::functional::linear(value, w_chunks[2]);
      }
    }
  } else {
    query_projected = q_proj_(query);
    key_projected = k_proj_(key);
    value_projected = v_proj_(value);
  }

  // Step 2. Split heads: (N, L, E_total) -> (N, nheads, L, d_head)
  query_projected = query_projected
                        .view({query_projected.size(0), query_projected.size(1),
                               nheads_, d_head_})
                        .transpose(1, 2);
  key_projected = key_projected
                      .view({key_projected.size(0), key_projected.size(1),
                             nheads_, d_head_})
                      .transpose(1, 2);
  value_projected = value_projected
                        .view({value_projected.size(0), value_projected.size(1),
                               nheads_, d_head_})
                        .transpose(1, 2);

  // Step 3. Scaled dot-product attention
  auto attn_output = at::native::scaled_dot_product_attention(
      query_projected, key_projected, value_projected, attn_mask, dropout_p_,
      is_causal);

  // (N, nheads, L, d_head) -> (N, L, nheads, d_head) -> (N, L, E_total)
  attn_output = attn_output.transpose(1, 2).reshape(
      {attn_output.size(0), attn_output.size(2), nheads_ * d_head_});

  // Step 4. Output projection
  return out_proj_(attn_output);
}
