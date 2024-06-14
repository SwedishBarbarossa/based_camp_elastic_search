import os
from concurrent import futures
from time import sleep

import grpc
import numpy as np
from grpc import StatusCode

from services.record_funcs import _calculate_checksum_of_str

from . import service_pb2, service_pb2_grpc

_AUTH_HEADER_KEY = "authorization"
GRPC_API_KEY = os.environ["GRPC_API_KEY"]
_AUTH_HEADER_VALUE = _calculate_checksum_of_str(GRPC_API_KEY or "test_token")


class SignatureValidationInterceptor(grpc.ServerInterceptor):
    def __init__(self):
        def abort(ignored_request, context):
            print("invalid")
            random_sleep_duration = float(np.random.normal(2, 0.5))
            sleep(random_sleep_duration)
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
        name: str = request.input_string
        split_name = name.split(" ")
        channel = split_name[1]
        video_id = split_name[2]
        array = np.array(request.array, dtype=np.float32)
        save_path = os.path.join(self.embeddings_dir, channel, video_id)
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, name + ".npy"), array)
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
