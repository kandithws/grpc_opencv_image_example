import grpc
import detection_proto.detection_pb2
import detection_proto.detection_pb2_grpc
import cv2
from detection_proto.cv_img_converter import to_cv2
from concurrent import futures
import time


class DetectionServicer(detection_proto.detection_pb2_grpc.DetectionServiceServicer):

    def ObjectDetection(self, image, context):
        # Non-streaming based
        # convert to CV2
        mat = to_cv2(image)
        cv2.imwrite('test_out.jpeg', mat)
        msg = detection_proto.detection_pb2.Detections()
        msg.detections.extend([])
        return msg


def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    detection_proto.detection_pb2_grpc.add_DetectionServiceServicer_to_server(DetectionServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print('Starting server at localhost:50051')
    try:
        while True:
            time.sleep(100)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    main()
