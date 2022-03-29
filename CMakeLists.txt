cmake_minimum_required(VERSION 3.10)

project(MathCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(Application application.cpp)

target_include_directories(Application
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )
