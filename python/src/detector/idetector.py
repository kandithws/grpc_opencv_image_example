from detection_proto.detection_pb2 import Detections


class IDetector:
    def detect_object(self, image) -> Detections:
        raise NotImplementedError()

    @classmethod
    def build(cls, detector_type): # TODO
        pass
