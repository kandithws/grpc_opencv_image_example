import cv2
import numpy as np
# DIRTY WRITE for now
from distutils.version import StrictVersion
from detector.idetector import IDetector
from detection_proto.detection_pb2 import Detections, Detection, Rect, Point2d


class CVDetector:
    def __init__(self, paths, config=None):
        if config is None:
            config = {}

        assert (StrictVersion(cv2.__version__) >= StrictVersion('3.4.2'))
        self.conf_th = None
        self.nms_th = None
        self.im_width = None
        self.im_height = None
        self.net = None
        self.classes = None

    def get_output_names(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def detect(self, image, draw_boxes=False, apply_nms=True, swapRB=True):
        # convert RGB to BGR
        # cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # blob = cv2.dnn.blobFromImage(cv_image, 1./255, (self.im_width, self.im_height), [0,0,0], swapRB=1, crop=False)
        blob = cv2.dnn.blobFromImage(image, 1. / 255, (self.im_width, self.im_height), [0, 0, 0], swapRB=swapRB, crop=False)
        self.net.setInput(blob)
        if self.net_output_names is not None:
            outs = self.net.forward(self.net_output_names)
        else:
            outs = self.net.forward()
        # remove bb with low confidences
        classIds, confidences, boxes_rect = self.post_process(image, outs)

        if apply_nms:
            classIds, confidences, boxes_rect = self.nms(classIds, confidences, boxes_rect)

        if draw_boxes:
            self.draw_boxes(image, classIds, confidences, boxes_rect)

        return classIds, confidences, boxes_rect  # TODO return boxes as standard format instead

    def draw_boxes(self, frame, classIds, confidences, boxes):
        assert (len(classIds) == len(confidences) and len(confidences) == len(boxes))
        for i in range(len(classIds)):
            self.draw_box(frame, classIds[i], confidences[i], boxes[i])

    def draw_box(self, frame, classId, conf, box):
        # Draw a bounding box.
        # print(box)             left  top          right            btm
        cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255))
        label = '%.2f' % conf
        # Get the label for the class name and its confidence
        if self.classes:
            assert (classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)
        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(box[1], labelSize[1])
        cv2.putText(frame, label, (box[0], top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    def post_process(self, frame, outs):
        raise NotImplementedError

    def nms(self, classIds, confidences, boxes):
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        filtered_indices = np.reshape(cv2.dnn.NMSBoxes(boxes, confidences, self.conf_th, self.nms_th), -1)
        filtered_classIds = np.take(classIds, filtered_indices, axis=0)
        filtered_confidences = np.take(confidences, filtered_indices, axis=0)
        filtered_boxes = np.take(boxes, filtered_indices, axis=0)
        return filtered_classIds, filtered_confidences, filtered_boxes


class DarknetCVDetector(CVDetector, IDetector):
    def __init__(self, paths, config=None):
        if config is None:
            config = {}

        CVDetector.__init__(self, paths, config)
        assert ('names' in paths)
        assert ('model_config' in paths)
        assert ('model_weight' in paths)
        with open(paths['names'], 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.conf_th = config['confidence'] if 'confidence' in config else 0.5
        self.nms_th = config['nms_threshold'] if 'nms_threshold' in config else 0.4
        self.im_width = config['im_width'] if 'im_width' in config else 416
        self.im_height = config['im_height'] if 'im_width' in config else 416

        self.net = cv2.dnn.readNetFromDarknet(paths['model_config'], paths['model_weight'])
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.net_output_names = self.get_output_names(self.net)

    def detect_object(self, image) -> Detections:
        (classIds, confidences, boxes_rect) = self.detect(image, swapRB=True)
        detections = Detections()
        ds = []
        
        for id, conf, box in zip(classIds, confidences, boxes_rect):
            d = Detection()
            d.confidence = conf
            d.label_id = id
            d.label = self.classes[id]
            # Doing like this due to protobuf
            d.box.tl.x = box[0]
            d.box.tl.y = box[1]
            d.box.br.x = box[0] + box[2]
            d.box.br.y = box[1] + box[3]
            ds.append(d)
        detections.detections.extend(ds)
        return detections


    # Remove the bounding boxes with low confidence using non-maxima suppression
    def post_process(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence >= self.conf_th:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    # Note: need to changed to left top to match OpenCV's Rect object
                    boxes.append([left, top, width, height])

        return classIds, confidences, boxes