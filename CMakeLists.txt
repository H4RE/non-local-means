cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./TEST_)

project(template)
add_executable(${PROJECT_NAME}
src/main.cpp
)

find_package(Halide REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Halide::Halide)

find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()