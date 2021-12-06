#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

# import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'contact_graspnet')
POINTNET_ROOT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'contact_graspnet/pointnet2')
PATH_TO_CHECKPOINT = os.path.join(ROOT_DIR, 'checkpoints/scene_test_2048_bs3_hor_sigma_001') #os.path.join(ROOT_DIR, 'checkpoints/ACRONYM_GAN/tf_output')  #os.path.join(ROOT_DIR, 'checkpoints/ACRONYM_Evaluator/tf_output')
SAVEDMODEL_PATH = os.path.join(PATH_TO_CHECKPOINT, "tftrt_model")
trained_checkpoint_prefix = os.path.join(PATH_TO_CHECKPOINT, 'model.ckpt-72072') #os.path.join(PATH_TO_CHECKPOINT, 'model-2241334') 

def load_ops_library():
    TF_OPS_DIR = os.path.join(POINTNET_ROOT_DIR, 'tf_ops')
    print(os.path.join(TF_OPS_DIR, 'sampling/tf_sampling_so.so'))
    sampling_module = tf.load_op_library(os.path.join(TF_OPS_DIR, 'sampling/tf_sampling_so.so'))
    grouping_module = tf.load_op_library(os.path.join(TF_OPS_DIR, 'grouping/tf_grouping_so.so'))
    interpolation_module = tf.load_op_library(os.path.join(TF_OPS_DIR, '3d_interpolation/tf_interpolate_so.so'))

def load_and_convert(path, precision):
    """ Load a saved model and convert it to FP32 or FP16. Return a converter """

    converter = trt.TrtGraphConverter(
        input_saved_model_dir=path,
        max_workspace_size_bytes=(1<32),
        precision_mode="FP16",
        maximum_cached_engines=100,
        minimum_segment_size=3,
        is_dynamic_op=True,
    )

    return converter

def load_and_convert_v2(path, precision):
    """ Load a saved model and convert it to FP32 or FP16. Return a converter """

    params = copy.deepcopy(trt.DEFAULT_TRT_CONVERSION_PARAMS)

    params = params._replace(
        precision_mode=(
            trt.TrtPrecisionMode.FP16
            if precision.lower() == "fp16" else
            trt.TrtPrecisionMode.FP32
        ),
        max_workspace_size_bytes=(1<<32),  # 8,589,934,592 bytes
        maximum_cached_engines=1,
        minimum_segment_size=3,
        is_dynamic_op=True,
        # allow_build_at_runtime=True
    )

    import pprint
    print("%" * 85)
    pprint.pprint(params)
    print("%" * 85)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=path,
        conversion_params=params,
    )

    return converter

if __name__ == '__main__':
    load_ops_library()

    converter = load_and_convert_v2(
        os.path.join(SAVEDMODEL_PATH),
        precision="fp16"
    )
    print(converter)
    trt_graph = converter.convert()
    print('Finish optimizing with TF-TRT!')

    converter.save(
        os.path.join(SAVEDMODEL_PATH, "converted")
    )
    print('TRT graph is saved!')
