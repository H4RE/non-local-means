cmake_minimum_required(VERSION 3.25)
project(non-local-means)

find_package(Halide REQUIRED)
add_halide_generator(nlm_halide_generator SOURCES ./src/non_local_means_halide_generator.cpp)
add_halide_library(non_local_means_halide FROM nlm_halide_generator)


enable_language("CUDA")

add_library(nlm-lib
STATIC 
./src/non_local_means.cpp
./src/non_local_means.cu
)

target_compile_features(nlm-lib PRIVATE cxx_std_17)
target_include_directories(nlm-lib PUBLIC "./include/")


find_package(OpenCV REQUIRED)
target_link_libraries(nlm-lib PRIVATE ${OpenCV_LIBS})

target_link_libraries(nlm-lib PRIVATE Halide::Halide Halide::ImageIO)
target_link_libraries(nlm-lib PUBLIC non_local_means_halide)


find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
