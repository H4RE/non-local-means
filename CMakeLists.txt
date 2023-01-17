cmake_minimum_required(VERSION 3.25)
project(main)
enable_language("CUDA")
add_executable(${PROJECT_NAME} "./main.cpp")
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)


find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

find_package(Halide REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Halide::Halide Halide::ImageIO)

add_subdirectory(./non-local-means)
target_link_libraries(${PROJECT_NAME} PRIVATE nlm-lib)