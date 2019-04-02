import cv2
from sys import argv
from detection_proto.cv_img_converter import from_cv2
import detection_proto.detection_pb2_grpc
import grpc


def main():
    with grpc.insecure_channel('0.0.0.0:50051') as channel:
        stub = detection_proto.detection_pb2_grpc.DetectionServiceStub(channel)
        msg = from_cv2(cv2.imread(argv[1]))
        response = stub.ObjectDetection(msg)
        print(type(response))
    print("--------DONE----------")


if __name__ == '__main__':
    if len(argv) != 2:
        print("Usage: client.py [input image]")
        exit(0)
    main()
