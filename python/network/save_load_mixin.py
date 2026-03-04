import torch
import torch.nn as nn
import os


class SaveLoadMixin:
    def save_weights(self, path):
        with open(path, "wb") as file:
            for name, param in self.named_parameters():
                tensor = param.detach().cpu().contiguous()

                name_len = len(name)
                file.write(name_len.to_bytes(8, byteorder="little"))
                file.write(name.encode("utf-8"))

                shape = tensor.size()
                ndims = len(shape)
                file.write(ndims.to_bytes(8, byteorder="little"))
                for dim in shape:
                    file.write(dim.to_bytes(8, byteorder="little"))

                num_elems = tensor.numel()
                file.write(tensor.numpy().tobytes())

            for name, buffer in self.named_buffers():
                tensor = buffer.detach().cpu().contiguous()

                name_len = len(name)
                file.write(name_len.to_bytes(8, byteorder="little"))
                file.write(name.encode("utf-8"))

                shape = tensor.size()
                ndims = len(shape)
                file.write(ndims.to_bytes(8, byteorder="little"))
                for dim in shape:
                    file.write(dim.to_bytes(8, byteorder="little"))

                num_elems = tensor.numel()
                file.write(tensor.numpy().tobytes())

    def load_weights(self, path):
        if not os.path.exists(path):
            raise RuntimeError("Failed to open file for reading")

        with open(path, "rb") as file:
            params_map = dict(self.named_parameters())
            buffers_map = dict(self.named_buffers())

            while True:
                name_len = int.from_bytes(file.read(8), byteorder="little")
                if not name_len:
                    break

                name = file.read(name_len).decode("utf-8")

                target_tensor = params_map.get(name)
                if target_tensor is None:
                    target_tensor = buffers_map.get(name)

                if target_tensor is None:
                    raise RuntimeError(
                        f"Parameter or buffer '{name}' not found in model"
                    )

                ndims = int.from_bytes(file.read(8), byteorder="little")
                shape = [
                    int.from_bytes(file.read(8), byteorder="little")
                    for _ in range(ndims)
                ]

                if len(shape) == 0:
                    shape = []

                expected_shape = target_tensor.shape

                if tuple(shape) != tuple(expected_shape):
                    raise RuntimeError(
                        f"Shape mismatch for {name}, expected {expected_shape}, got {shape}"
                    )

                num_elems = 1
                for dim in shape:
                    num_elems *= dim
                data = file.read(num_elems * target_tensor.element_size())
                tensor_data = torch.frombuffer(data, dtype=target_tensor.dtype).reshape(
                    shape
                )

                with torch.no_grad():
                    target_tensor.copy_(tensor_data)
