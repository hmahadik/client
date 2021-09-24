#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
from PIL import Image
import os
import sys
import struct
import cv2

import grpc

from tritonclient.grpc import service_pb2, service_pb2_grpc, InferResult
import tritonclient.grpc.model_config_pb2 as mc
import time

from timeit import default_timer as timer

FLAGS = None

LABELS = [
"person",
"bicycle",
"car",
"motorcycle",
"airplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"street sign",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"hat",
"backpack",
"umbrella",
"shoe",
"eye glasses",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"plate",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"couch",
"potted plant",
"bed",
"mirror",
"dining table",
"window",
"desk",
"toilet",
"door",
"tv",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"blender",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush"
]

def class_idx_to_label(idx):
    return LABELS[idx] if idx < len(LABELS) else None

def model_dtype_to_np(model_dtype):
    if model_dtype == "BOOL":
        return bool
    elif model_dtype == "INT8":
        return np.int8
    elif model_dtype == "INT16":
        return np.int16
    elif model_dtype == "INT32":
        return np.int32
    elif model_dtype == "INT64":
        return np.int64
    elif model_dtype == "UINT8":
        return np.uint8
    elif model_dtype == "UINT16":
        return np.uint16
    elif model_dtype == "FP16":
        return np.float16
    elif model_dtype == "FP32":
        return np.float32
    elif model_dtype == "FP64":
        return np.float64
    elif model_dtype == "BYTES":
        return np.dtype(object)
    return None


def deserialize_bytes_tensor(encoded_tensor):
    strs = list()
    offset = 0
    val_buf = encoded_tensor
    while offset < len(val_buf):
        l = struct.unpack_from("<I", val_buf, offset)[0]
        offset += 4
        sb = struct.unpack_from("<{}s".format(l), val_buf, offset)[0]
        offset += l
        strs.append(sb)
    return (np.array(strs, dtype=np.object_))


def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))
    if len(model_metadata.outputs) not in [1,4]:
        raise Exception("expecting 1 or 4 outputs, got {}".format(
            len(model_metadata.outputs)))

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    # Model input must have 3 dims, either CHW or HWC (not counting
    # the batch dimension), either CHW or HWC
    if (len(input_metadata.shape) == 4 and input_metadata.shape[0] == 1):
        model_config.max_batch_size = 1
    input_batch_dim = model_config.max_batch_size > 0
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_input_dims, model_metadata.name,
                   len(input_metadata.shape)))

    if type(input_config.format) == str:
        FORMAT_ENUM_TO_INT = dict(mc.ModelInput.Format.items())
        input_config.format = FORMAT_ENUM_TO_INT[input_config.format]
    
    if input_config.format == mc.ModelInput.FORMAT_NONE:
        input_config.format = mc.ModelInput.FORMAT_NHWC

    if ((input_config.format != mc.ModelInput.FORMAT_NCHW) and
        (input_config.format != mc.ModelInput.FORMAT_NHWC)):
        raise Exception("unexpected input format " +
                        mc.ModelInput.Format.Name(input_config.format) +
                        ", expecting " +
                        mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW) +
                        " or " +
                        mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC))

    if input_config.format == mc.ModelInput.FORMAT_NHWC:
        h = input_metadata.shape[1 if input_batch_dim else 0]
        w = input_metadata.shape[2 if input_batch_dim else 1]
        c = input_metadata.shape[3 if input_batch_dim else 2]
    else:
        c = input_metadata.shape[1 if input_batch_dim else 0]
        h = input_metadata.shape[2 if input_batch_dim else 1]
        w = input_metadata.shape[3 if input_batch_dim else 2]

    return (model_config.max_batch_size, input_metadata.name,
            model_metadata.outputs, c, h, w, input_config.format,
            input_metadata.datatype)

def preprocess(img, format, dtype, c, h, w, scaling):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    # np.set_printoptions(threshold='nan')
    img = Image.fromarray(img)
    
    if c == 1:
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    resized_img = img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    npdtype = model_dtype_to_np(dtype)
    typed = resized.astype(npdtype)

    if (scaling == 'INCEPTION') or (npdtype == np.float32):
        scaled = (typed / 127.5) - 1
    elif scaling == 'VGG':
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=npdtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
    else:
        scaled = typed

    # Swap to CHW if necessary
    if format == mc.ModelInput.FORMAT_NCHW:
        ordered = np.transpose(scaled, (2, 0, 1))
    else:
        ordered = scaled

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered


def postprocess(results, output_config, batch_size):
    """
    Post-process results to show detections.
    """

    # classifications
    if len(output_config) == 1:
        classes = results.as_numpy(output_config[0].name)
        print(classes)
        return

    # detections
    detection_bboxes = results.as_numpy(output_config[0].name)[0]
    detection_classes = results.as_numpy(output_config[1].name)[0]
    detection_probs = results.as_numpy(output_config[2].name)[0]
    num_detections = int(results.as_numpy(output_config[3].name)[0])
    
    detected_objects = []
    print("Detections:")
    for i in range(num_detections):
        if detection_probs[i] > 0.25:
            detection_class_idx = int(detection_classes[i])
            detection_class_label = class_idx_to_label(detection_class_idx)
            detected_objects.append(
                dict(
                    cls=detection_class_label,
                    bbox=detection_bboxes[i],
                    prob=detection_probs[i]
                )
            )
            print(f"  {detection_class_label} ({round(detection_probs[i]*100.0, 1)}%)")


def requestGenerator(input_name, output_config, c, h, w, format, dtype, FLAGS):
    request = service_pb2.ModelInferRequest()
    request.model_name = FLAGS.model_name
    request.model_version = FLAGS.model_version
    
    outputs = []
    for output in output_config:
        o = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
        o.name = output.name
        outputs.append(o)
    request.outputs.extend(outputs)

    input = service_pb2.ModelInferRequest().InferInputTensor()
    input.name = input_name
    input.datatype = dtype
    if format == mc.ModelInput.FORMAT_NHWC:
        input.shape.extend([FLAGS.batch_size, h, w, c])
    else:
        input.shape.extend([FLAGS.batch_size, c, h, w])

    # Preprocess image into input data according to model requirements
    # Preprocess the images into input data according to model
    # requirements
    image_data = []
    
    cap = cv2.VideoCapture()
    cap.open(FLAGS.image_filename)
    assert(cap.isOpened())

    # Send requests of FLAGS.batch_size images. If the number of
    # images isn't an exact multiple of FLAGS.batch_size then just
    # start over with the first images until the batch is filled.
    image_idx = 0
    last_request = False
    while not last_request:
        retval, img = cap.read(cv2.IMREAD_UNCHANGED)
        assert(retval)
        input_bytes = preprocess(img, format, dtype, c, h, w, FLAGS.scaling).tobytes()
        request.ClearField("inputs")
        request.ClearField("raw_input_contents")
        request.inputs.extend([input])
        request.raw_input_contents.extend([input_bytes])
        yield request


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-a',
                        '--async',
                        dest="async_set",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use asynchronous inference API')
    parser.add_argument('--streaming',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use streaming inference API')
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        required=True,
                        help='Name of model')
    parser.add_argument(
        '-x',
        '--model-version',
        type=str,
        required=False,
        default="",
        help='Version of model. Default is to use latest version.')
    parser.add_argument('-b',
                        '--batch-size',
                        type=int,
                        required=False,
                        default=1,
                        help='Batch size. Default is 1.')
    parser.add_argument('-c',
                        '--classes',
                        type=int,
                        required=False,
                        default=1,
                        help='Number of class results to report. Default is 1.')
    parser.add_argument(
        '-s',
        '--scaling',
        type=str,
        choices=['NONE', 'INCEPTION', 'VGG'],
        required=False,
        default='NONE',
        help='Type of scaling to apply to image pixels. Default is NONE.')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('image_filename',
                        type=str,
                        nargs='?',
                        default=None,
                        help='Input image / Input folder.')
    FLAGS = parser.parse_args()

    # Create gRPC stub for communicating with the server
    channel = grpc.insecure_channel(FLAGS.url)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    metadata_request = service_pb2.ModelMetadataRequest(
        name=FLAGS.model_name, version=FLAGS.model_version)
    metadata_response = grpc_stub.ModelMetadata(metadata_request)

    config_request = service_pb2.ModelConfigRequest(name=FLAGS.model_name,
                                                    version=FLAGS.model_version)
    config_response = grpc_stub.ModelConfig(config_request)

    max_batch_size, input_name, output_config, c, h, w, format, dtype = parse_model(
        metadata_response, config_response.config)

    # Send requests of FLAGS.batch_size images. If the number of
    # images isn't an exact multiple of FLAGS.batch_size then just
    # start over with the first images until the batch is filled.
    requests = []
    responses = []
    result_filenames = []

    # Send request
    frameNumber = 1
    start = timer()
    for request in requestGenerator(input_name, output_config, c, h, w,
                                    format, dtype, FLAGS):
        preProcessTime = timer()
        response = grpc_stub.ModelInfer(request)
        result = InferResult(response)
        inferenceTime = timer()
        os.system('clear')
        postprocess(result, output_config, 1)
        postProcessTime = timer()
        frameNumber += 1
        end = timer()
        totalTime = end - start
        print(f"   Pre-process : {round((preProcessTime-start)*1000,1)} ms")
        print(f"   Inference   : {round((inferenceTime-preProcessTime)*1000,1)} ms")
        print(f"   Post-process: {round((postProcessTime-inferenceTime)*1000,1)} ms")
        print(f"** Total : {round(totalTime*1000,1)} ms ({round(1.0/totalTime,1)} inf/sec)")
        start = timer()
        
""" 
To run: 
python3 client/src/python/examples/grpc_image_ssd_client.py -m ssd_mobilenet_v2_coco --streaming http://192.168.1.16/mjpg/video.mjpg 

Output:
Detections:
  potted plant (96.6%)
   Pre-process : 68.0 ms
   Inference   : 753.3 ms
   Post-process: 7.9 ms
** Total : 829.3 ms (1.2 inf/sec)
"""
