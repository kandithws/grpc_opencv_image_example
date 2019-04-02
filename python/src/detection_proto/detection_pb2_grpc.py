# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import detection_proto.detection_pb2 as detection__pb2


class DetectionServiceStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.ObjectDetection = channel.unary_unary(
        '/DetectionService/ObjectDetection',
        request_serializer=detection__pb2.Image.SerializeToString,
        response_deserializer=detection__pb2.Detections.FromString,
        )


class DetectionServiceServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def ObjectDetection(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_DetectionServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'ObjectDetection': grpc.unary_unary_rpc_method_handler(
          servicer.ObjectDetection,
          request_deserializer=detection__pb2.Image.FromString,
          response_serializer=detection__pb2.Detections.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'DetectionService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
