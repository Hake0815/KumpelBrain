cmake_minimum_required(VERSION 4.2.3)

project(kumpel_embedding)

set(Torch_DIR "/home/felix/ai/KumpelBrain/.venv/lib/python3.13/site-packages/torch/share/cmake/Torch")
# set(Torch_DIR "$ENV{HOME}/.lib/libtorch/share/cmake/Torch")
# set(CMAKE_PREFIX_PATH "$ENV{HOME}/.lib/libtorch")
set(Python_FIND_VIRTUALENV ONLY)

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(kumpel_embedding
    network/src/PositionalEmbedding.cpp
    network/src/MultiHeadAttention.cpp
    network/src/NormalizedLinear.cpp
    network/src/bindings.cpp
)

target_include_directories(kumpel_embedding PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kumpel_embedding PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

set_property(TARGET kumpel_embedding PROPERTY CXX_STANDARD 17)