cmake_minimum_required(VERSION 3.20)

project(kumpel_embedding)

set(Python_FIND_VIRTUALENV ONLY)

find_package(Torch REQUIRED)
find_package(Protobuf REQUIRED)
find_package(absl CONFIG REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})

find_package(pybind11 CONFIG REQUIRED)

set(KUMPEL_CORE_SOURCES
    network/src/AttentionUtils.cpp
    network/src/AbilityEmbedding.cpp
    network/src/AttackDataEmbedding.cpp
    network/src/AttackEmbedding.cpp
    network/src/CardAmountDataEmbedding.cpp
    network/src/CardEmbedding.cpp
    network/src/CardStateEmbedding.cpp
    network/src/ConditionEmbedding.cpp
    network/src/DiscardDataEmbedding.cpp
    network/src/EnergyTypeEmbedding.cpp
    network/src/FilterConditionEmbedding.cpp
    network/src/FilterEmbedding.cpp
    network/src/InstructionDataEmbedding.cpp
    network/src/InstructionEmbedding.cpp
    network/src/PositionalEmbedding.cpp
    network/src/Nesting.cpp
    network/src/TensorUtils.cpp
    network/src/PlayerTargetDataEmbedding.cpp
    network/src/ReturnToDeckTypeDataEmbedding.cpp
    network/src/MultiHeadAttention.cpp
    network/src/NormalizedLinear.cpp
    network/src/SharedEmbeddingHolder.cpp
    network/src/serialization/gamecore_serialization.pb.cc
)

pybind11_add_module(kumpel_embedding
    ${KUMPEL_CORE_SOURCES}
    network/src/bindings.cpp
)

target_include_directories(kumpel_embedding BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kumpel_embedding PRIVATE
    ${Protobuf_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    protobuf::libprotobuf
)

set_property(TARGET kumpel_embedding PROPERTY CXX_STANDARD 17)

add_executable(kumpel_embedding_bench
    network/benchmark_main.cpp
    ${KUMPEL_CORE_SOURCES}
)

target_include_directories(kumpel_embedding_bench BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kumpel_embedding_bench PRIVATE
    ${Protobuf_LIBRARIES}
    ${TORCH_LIBRARIES}
    protobuf::libprotobuf
    absl::absl_check
    absl::absl_log
)

set_property(TARGET kumpel_embedding_bench PROPERTY CXX_STANDARD 17)