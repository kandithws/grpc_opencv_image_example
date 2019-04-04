import cv2
from detection_proto.detection_pb2 import Image
import numpy as np


def from_cv2(img):
    msg = Image()
    msg.height, msg.width = img.shape[:2]
    msg.data = bytes(img.reshape(-1))
    if img.shape[2] == 1:
        msg.encoding = "gray"
    else:
        msg.encoding = "bgr8"

    return msg


def to_cv2(msg):
    width = msg.width
    height = msg.height   
    np_data = np.frombuffer(msg.data, dtype=np.uint8)
    print(np_data.shape)
    if msg.encoding == "bgr8" or msg.encoding == "rgb8":
        img = np_data.reshape((height, width, 3))
    else:
        img = np_data.reshape((height, width, 1))
    return img


if __name__ == '__main__':
    img = cv2.imread('/home/kandithws/profile.jpeg')
    msg = from_cv2(img)
    out_img = to_cv2(msg)
    cv2.imwrite('out.jpg', out_img)
    # print(msg.data)
