import os
from concurrent import futures

import grpc
import numpy as np
from grpc import StatusCode

from . import service_pb2, service_pb2_grpc

_AUTH_HEADER_KEY = "authorization"
GRPC_API_KEY = os.environ["GRPC_API_KEY"]
_AUTH_HEADER_VALUE = GRPC_API_KEY or "test_token"


class SignatureValidationInterceptor(grpc.ServerInterceptor):
    def __init__(self):
        def abort(ignored_request, context):
            print("invalid")
            context.abort(StatusCode.UNAUTHENTICATED, "Invalid signature")

        self._abort_handler = grpc.unary_unary_rpc_method_handler(abort)

    def intercept_service(self, continuation, handler_call_details):
        expected_metadata = (_AUTH_HEADER_KEY, _AUTH_HEADER_VALUE)
        if expected_metadata in handler_call_details.invocation_metadata:
            return continuation(handler_call_details)
        else:
            return self._abort_handler


class StringAndArrayService(service_pb2_grpc.StringAndArrayServiceServicer):
    def __init__(self, embeddings_dir: str, callback=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback = callback
        self.embeddings_dir = embeddings_dir

    def ProcessData(self, request, context):
        # Example processing: sum of the array elements + string length
        name = request.input_string
        array = np.array(request.array, dtype=np.float32)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        np.save(os.path.join(self.embeddings_dir, name + ".npy"), array)
        if self.callback is not None:
            self.callback(name)
        result = f"Array {request.input_string} saved"
        return service_pb2.DataResponse(result=result)


def serve(embeddings_dir: str, callback=None):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=(SignatureValidationInterceptor(),),
    )
    service_pb2_grpc.add_StringAndArrayServiceServicer_to_server(
        StringAndArrayService(embeddings_dir, callback), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()
