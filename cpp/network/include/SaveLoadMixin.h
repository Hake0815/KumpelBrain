#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

/*
 * CRTP mixin for torch::nn::Module
 * Adds save_model / load_model to any derived class.
 */
template <typename Derived> struct SaveLoadMixin {

  void save_weights(const std::string &path) {
    const auto &self = static_cast<const Derived &>(*this);

    std::ofstream file(path, std::ios::binary);
    if (!file)
      throw std::runtime_error("Failed to open file for writing");

    for (const auto &pair : self.named_parameters()) {
      const std::string &name = pair.key();
      const torch::Tensor &param = pair.value();
      torch::Tensor tensor = param.cpu().contiguous();

      int64_t name_len = name.size();
      file.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
      file.write(name.data(), name_len);

      auto shape = tensor.sizes();
      int64_t ndims = shape.size();
      file.write(reinterpret_cast<const char *>(&ndims), sizeof(ndims));
      for (auto dim : shape) {
        file.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
      }

      int64_t num_elems = tensor.numel();
      file.write(reinterpret_cast<const char *>(tensor.data_ptr()),
                 num_elems * tensor.element_size());
    }

    for (const auto &pair : self.named_buffers()) {
      const std::string &name = pair.key();
      const torch::Tensor &param = pair.value();
      torch::Tensor tensor = param.cpu().contiguous();

      int64_t name_len = name.size();
      file.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
      file.write(name.data(), name_len);

      auto shape = tensor.sizes();
      int64_t ndims = shape.size();
      file.write(reinterpret_cast<const char *>(&ndims), sizeof(ndims));
      for (auto dim : shape) {
        file.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
      }

      int64_t num_elems = tensor.numel();
      file.write(reinterpret_cast<const char *>(tensor.data_ptr()),
                 num_elems * tensor.element_size());
    }
  }
  void load_weights(const std::string &path) {
    auto &self = static_cast<Derived &>(*this);
    std::ifstream file(path, std::ios::binary);
    if (!file)
      throw std::runtime_error("Failed to open file for reading");

    std::unordered_map<std::string, torch::Tensor> params_map;
    for (auto &pair : self.named_parameters()) {
      params_map[pair.key()] = pair.value();
    }

    std::unordered_map<std::string, torch::Tensor> buffers_map;
    for (auto &pair : self.named_buffers()) {
      buffers_map[pair.key()] = pair.value();
    }

    while (file.peek() != EOF) {
      int64_t name_len;
      file.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
      std::string name(name_len, '\0');
      file.read(name.data(), name_len);

      torch::Tensor *target_tensor = nullptr;
      auto it_param = params_map.find(name);
      if (it_param != params_map.end()) {
        target_tensor = &it_param->second;
      } else {
        auto it_buf = buffers_map.find(name);
        if (it_buf != buffers_map.end()) {
          target_tensor = &it_buf->second;
        } else {
          throw std::runtime_error("Parameter or buffer " + name +
                                   " not found in model");
        }
      }

      torch::Tensor &param = *target_tensor;

      int64_t ndims;
      file.read(reinterpret_cast<char *>(&ndims), sizeof(ndims));
      std::vector<int64_t> shape(ndims);
      for (int i = 0; i < ndims; ++i) {
        file.read(reinterpret_cast<char *>(&shape[i]), sizeof(shape[i]));
      }

      auto expected_shape = param.sizes();
      if (shape != expected_shape) {
        throw std::runtime_error("Shape mismatch for " + name);
      }

      int64_t num_elems = 1;
      for (auto dim : shape)
        num_elems *= dim;
      file.read(reinterpret_cast<char *>(param.data_ptr()),
                num_elems * param.element_size());
    }
  }
};