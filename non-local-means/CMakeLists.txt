cmake_minimum_required(VERSION 3.25)
project(non-local-means)

file(GLOB cpp_files ./src/*.cpp)
file(GLOB cuda_files ./src/*.cu)

enable_language("CUDA")

add_library(nlm-lib STATIC ${cpp_files} ${cuda_files})
target_compile_features(nlm-lib PRIVATE cxx_std_17)
target_include_directories(nlm-lib PUBLIC "./include/")

find_package(OpenCV REQUIRED)
target_link_libraries(nlm-lib PRIVATE ${OpenCV_LIBS})

find_package(Halide REQUIRED)
target_link_libraries(nlm-lib PRIVATE Halide::Halide Halide::ImageIO)


find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")