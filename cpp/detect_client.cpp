//
// Created by kandithws on 2/4/2562.
//

#include <iostream>
#include <grpcpp/grpcpp.h>
#include <opencv2/opencv.hpp>
#include "proto-src/detection.grpc.pb.h"

class DetectionClient {
  public:
    DetectionClient(std::shared_ptr<grpc::Channel> channel)
    : stub_(DetectionService::NewStub(channel)) {}

    bool detectObject(const cv::Mat& img, std::vector<Detection>& out){
        Detections response;
        auto req_img = toImageRequest(img);
        grpc::ClientContext context;
        auto st = stub_->ObjectDetection(&context, req_img, &response);
        std::cout << "I AM HERE!" << std::endl;

        if (st.ok()){

            // TODO -- add detection to vector
            return true;
        }
        else {
            std::cout << "ERROR! " << std::endl;
            std::cout << st.error_message() << std::endl;
            return false;
        }
    }

    Image toImageRequest(const cv::Mat& img, std::string encoding="bgr8") {
        auto img_msg = Image();
        img_msg.set_encoding(encoding);
        size_t size = img.total() * img.elemSize();
        char* bytes = new char[size];
        std::memcpy(bytes, img.data, size * sizeof(char));
        img_msg.set_data(bytes);
        delete[] bytes;
        img_msg.set_height(img.cols);
        img_msg.set_width(img.rows);
        return img_msg;
    };

  private:
    std::unique_ptr<DetectionService::Stub> stub_;
};

int main(int argc, char** argv){
    DetectionClient image_client(grpc::CreateChannel(
            "localhost:50051", grpc::InsecureChannelCredentials()));

    if (argc != 2){
        std::cout << "Usage: image_client [image_file]" << std::endl;
        return -1;
    }
    cv::Mat img = cv::imread(argv[1]);
    std::vector<Detection> out;
    if(image_client.detectObject(img, out)){
        std::cout << "DONE!!!" << std::endl;
    }
    else {
        std::cout << "Not connected" << std::endl;
    }
    return 0;
}

