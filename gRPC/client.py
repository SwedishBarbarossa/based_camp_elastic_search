import os

import grpc
import numpy as np
import numpy.typing as npt

from . import service_pb2, service_pb2_grpc

HOST_IP = os.environ["HOST_IP"]
GRPC_PORT = os.environ["GRPC_PORT"]

_AUTH_HEADER_KEY = "authorization"
GRPC_API_KEY = os.environ["GRPC_API_KEY"]
_AUTH_HEADER_VALUE = GRPC_API_KEY or "test_token"


class AuthInterceptor(grpc.UnaryUnaryClientInterceptor):
    def __init__(self, token):
        self.token = token

    def intercept_unary_unary(self, continuation, client_call_details, request):
        metadata = []
        if client_call_details.metadata is not None:
            metadata = list(client_call_details.metadata)
        metadata.append((_AUTH_HEADER_KEY, self.token))
        client_call_details = client_call_details._replace(metadata=metadata)
        return continuation(client_call_details, request)


def send_embedding(filename: str, array: npt.NDArray[np.float32]) -> str:
    channel = grpc.insecure_channel(f"{HOST_IP}:{GRPC_PORT}")
    intercept_channel = grpc.intercept_channel(
        channel, AuthInterceptor(_AUTH_HEADER_VALUE)
    )
    stub = service_pb2_grpc.StringAndArrayServiceStub(intercept_channel)
    response = stub.ProcessData(
        service_pb2.DataRequest(input_string=filename, array=array)
    )
    return "Client received: " + response.result
