cmake_minimum_required(VERSION 4.2.3)

project(kumpel_embedding)

set(CMAKE_PREFIX_PATH "$ENV{HOME}/.lib/libtorch")
set(Torch_DIR "$ENV{HOME}/.lib/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(${PROJECT_NAME} STATIC
  network/src/PositionalEmbedding.cpp
  network/src/positional_embedding_bindings.cpp
  network/src/settings.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)