# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import service_pb2 as service__pb2


class StringAndArrayServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ProcessData = channel.unary_unary(
            "/transcript_encoding.StringAndArrayService/ProcessData",
            request_serializer=service__pb2.DataRequest.SerializeToString,
            response_deserializer=service__pb2.DataResponse.FromString,
        )


class StringAndArrayServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ProcessData(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_StringAndArrayServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "ProcessData": grpc.unary_unary_rpc_method_handler(
            servicer.ProcessData,
            request_deserializer=service__pb2.DataRequest.FromString,
            response_serializer=service__pb2.DataResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "transcript_encoding.StringAndArrayService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class StringAndArrayService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ProcessData(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/transcript_encoding.StringAndArrayService/ProcessData",
            service__pb2.DataRequest.SerializeToString,
            service__pb2.DataResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
