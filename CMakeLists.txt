cmake_minimum_required(VERSION 3.24)
project(packed_swiglu_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(packed_swiglu_bench csrc/packed_swiglu_bench.cu)
target_compile_options(packed_swiglu_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>)

