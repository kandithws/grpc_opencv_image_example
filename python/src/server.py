import grpc
import detection_proto.detection_pb2
import detection_proto.detection_pb2_grpc
import cv2
from detection_proto.cv_img_converter import to_cv2
from concurrent import futures
import time
from detector.cv_detector import DarknetCVDetector


class DetectionServicer(detection_proto.detection_pb2_grpc.DetectionServiceServicer):
    def __init__(self):
        self._detector = DarknetCVDetector({
            'names': '/home/kandithws/ait_workspace/thesis_ml_ws/models/darknet-coco/coco.names',
            'model_config': '/home/kandithws/ait_workspace/thesis_ml_ws/models/darknet-coco/yolov3.cfg',
            'model_weight': '/home/kandithws/ait_workspace/thesis_ml_ws/models/darknet-coco/yolov3.weights'
        })

    def ObjectDetection(self, image, context):
        # Non-streaming based
        # convert to CV2
        mat = to_cv2(image)
        msg = self._detector.detect_object(mat)
        cv2.imwrite('detection_out.jpeg', mat)
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
