cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB cpp_files "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB cu_files "${CMAKE_SOURCE_DIR}/src/*.cu")

enable_language("CUDA")
include_directories("${CMAKE_SOURCE_DIR}/include/")

project(template)
message(STATUS ${cpp_files})
add_executable(${PROJECT_NAME} ${cpp_files} ${cu_files})

find_package(Halide REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Halide::Halide Halide::ImageIO)


find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
