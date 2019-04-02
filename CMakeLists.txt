project(grpc-cmake-example)
cmake_minimum_required(VERSION 3.2)

add_compile_options(-std=c++11)

# Cmake find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Protobuf REQUIRED)
find_package(GRPC REQUIRED)
find_package(OpenCV 3 REQUIRED)

set(PROTOS
        ${CMAKE_CURRENT_SOURCE_DIR}/protos/detection.proto
        )

set(PROTO_SRC_DIR ${PROJECT_SOURCE_DIR}/cpp/proto-src)
file(MAKE_DIRECTORY ${PROTO_SRC_DIR})
include_directories(${PROTO_SRC_DIR})

# for auto generation
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_SRC_DIR} ${PROTOS})
grpc_generate_cpp(GRPC_SRCS GRPC_HDRS ${PROTO_SRC_DIR} ${PROTOS})

# Building server
add_executable(image_client
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/detect_client.cpp
        ${PROTO_SRCS}
        ${GRPC_SRCS})
target_link_libraries(image_client
        gRPC::grpc++_reflection
        protobuf::libprotobuf
        ${OpenCV_LIBS}
        )