cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(mosquitopp_client LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
#find_package(OpenCV REQUIRED)

include_directories(/usr/include)
link_directories(/usr/lib/aarch64-linux-gnu)

add_executable(mqttExample main.cpp )
target_link_libraries(mqttExample mosquittopp mosquitto) # libbar.so is found in ${CMAKE_SOURCE_DIR}/lib


