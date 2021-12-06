#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################
### Execute in TF1 ###
######################
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import horovod.tensorflow as hvd  # Necessary to register Horovod OPs that may exists

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.saved_model import tag_constants
from tensorflow.tools.graph_transforms import TransformGraph

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'contact_graspnet')
POINTNET_ROOT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'contact_graspnet/pointnet2')
PATH_TO_CHECKPOINT  = os.path.join(ROOT_DIR, 'checkpoints/cgnet_checkpoints') 
trained_checkpoint_prefix = os.path.join(PATH_TO_CHECKPOINT, 'cg-net-1') 

def load_ops_library():
    TF_OPS_DIR = os.path.join(POINTNET_ROOT_DIR, 'tf_ops')
    print(os.path.join(TF_OPS_DIR, 'sampling/tf_sampling_so.so'))
    sampling_module = tf.load_op_library(os.path.join(TF_OPS_DIR, 'sampling/tf_sampling_so.so'))
    grouping_module = tf.load_op_library(os.path.join(TF_OPS_DIR, 'grouping/tf_grouping_so.so'))
    interpolation_module = tf.load_op_library(os.path.join(TF_OPS_DIR, '3d_interpolation/tf_interpolate_so.so'))

def export_graph_from_chkpts():
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        load_ops_library()
    
        # Restore from checkpoint
        print(trained_checkpoint_prefix + '.meta')
        loader = tf.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
        loader.restore(sess, trained_checkpoint_prefix)

        # save the graph
        tf.train.write_graph(sess.graph_def, PATH_TO_CHECKPOINT, 'graph.pbtxt') 

    # Write the graph as events
    train_writer = tf.summary.FileWriter(PATH_TO_CHECKPOINT)
    train_writer.add_graph(sess.graph)


def optimize_graph_for_inference(model_dir, input_node_names, output_node_names):
    input_graph_path = os.path.join(model_dir, 'graph.pbtxt')
    input_checkpoint = tf.train.latest_checkpoint(model_dir)

    input_binary = False
    clear_devices = True

    print("Loading Graph `{}` ...".format(input_graph_path))
    print("Loading Checkpoint `{}` ...".format(input_checkpoint))

    load_ops_library()

    print("Freezing Graph ...")
    # https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/python/tools/freeze_graph.py#L286-L301
    frozen_graph_def = freeze_graph.freeze_graph(
        input_graph=input_graph_path,  # A `GraphDef` file to load.
        input_saver="",  #  A TensorFlow Saver file.
        input_binary=input_binary,  #  A Bool. True means input_graph is .pb, False indicates .pbtxt.
        input_checkpoint=input_checkpoint,
        output_node_names=",".join(output_node_names),  # The name(s) of the output nodes, comma separated.
        restore_op_name="",  # Unused
        filename_tensor_name="",  # Unused
        output_graph=os.path.join(PATH_TO_CHECKPOINT, 'frozen_saved_model.pb'),  # String where to write the frozen `GraphDef`.
        clear_devices=clear_devices,  #  A Bool whether to remove device specifications.
        initializer_nodes="",  # Comma separated list of initializer nodes to run before freezing.
        variable_names_whitelist="",  # The set of variable names to convert (optional, bydefault, all variables are converted)
        variable_names_blacklist="",  # The set of variable names to omit converting to constants (optional)
        input_meta_graph=None,  # A `MetaGraphDef` file to load (optional).
        input_saved_model_dir=None,  # Path to the dir with TensorFlow 'SavedModel' file and variables (optional).
        saved_model_tags=tag_constants.SERVING,  # Group of comma separated tag(s) of the MetaGraphDef to load, in string format.
        checkpoint_version=saver_pb2.SaverDef.V2  # Tensorflow variable file format (saver_pb2.SaverDef.V1 or saver_pb2.SaverDef.V2
    )

    print("Optimizing Graph for Inference ...")
    optimized_frozen_graph = optimize_for_inference_lib.optimize_for_inference(
            frozen_graph_def,
            input_node_names,  # an array of the input node(s)
            output_node_names,  # an array of output nodes
            tf.float32.as_datatype_enum
    )

    
    # trt_engine_ops = len([1 for n in optimized_frozen_graph.node if str(n.op)=='TRTEngineOp'])
    # print("numb. of trt_engine_ops in trt_graph", trt_engine_ops)

    transforms = [
        'remove_nodes(op=Identity)',
        'merge_duplicate_nodes',
        'strip_unused_nodes',
        'fold_constants(ignore_errors=true)',
        'fold_batch_norms'
    ]

    print("Applying Graph Transformations ...")
    return TransformGraph(
        optimized_frozen_graph,
        input_node_names,  # an array of the input node(s)
        output_node_names,  # an array of output nodes
        transforms
    )

def convert_graph_def_to_saved_model(
    export_dir,
    graphdef_file,
    input_nodes=None,
    output_nodes=None
):
  if tf.io.gfile.exists(export_dir):
    tf.io.gfile.rmtree(export_dir)

  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graphdef_file, name='')

    print('Exporting Optimized SavedModel to Disk: `{}` ...'.format(export_dir))
    tf.saved_model.simple_save(
        sess,
        export_dir,
        inputs={
            name: sess.graph.get_tensor_by_name('{}:0'.format(node_name))
            for name, node_name in input_nodes
        },
        outputs={
            name: sess.graph.get_tensor_by_name('{}:0'.format(node_name))
            for name, node_name in output_nodes
        }
    )
    print('Optimized graph converted to SavedModel!')


if __name__ == "__main__":
    export_graph_from_chkpts()         # run this if the graph is not existed
    tftrt_ready_savedmodel_dir = os.path.join(PATH_TO_CHECKPOINT, "tftrt_model")

    try:
        os.makedirs(tftrt_ready_savedmodel_dir)
    except FileExistsError:
        pass

    # VAE
    # input_nodes = [
    #     ("vae_pc", "sample_pc_input"),
    #     ("vae_pred/samples", "sample_latent_input"),
    # ]
    # output_nodes = [
    #     ("fc_q", "vae/decoder/l2_normalize"),       
    #     ("fc_t", "vae/decoder/fc_t/BiasAdd"),      
    #     ("fc_conf", "vae/decoder/Sigmoid"),
    # ]

    # Evaluator
    input_nodes = [
        ("evaluator_pc", "evaluator_pc_input"),
        ("evaluator_grasp_eulers_0", "sample_ratation_0"),
        ("evaluator_grasp_eulers_1", "sample_ratation_1"),
        ("evaluator_grasp_eulers_2", "sample_ratation_2"),
        ("evaluator_grasp_translations", "sample_translation"),
    ]
    output_nodes = [
        ("tf_success_softmax", "evaluator/Slice_1"),
        ("fc_collision", "evaluator/fc_collision/BiasAdd"),
        ("fc_collision_confidence", "evaluator/Sigmoid"),
        # ("fc_collision_gradient", "evaluator/gradients/evaluator/fc_collision/MatMul_grad/MatMul"),
    ]

    inference_graph_def = optimize_graph_for_inference(
        model_dir=os.path.join(PATH_TO_CHECKPOINT, 'tf_output'),
        input_node_names=[node_name for _, node_name in input_nodes],
        output_node_names=[node_name for _, node_name in output_nodes]
    )

    convert_graph_def_to_saved_model(
        tftrt_ready_savedmodel_dir,
        graphdef_file=inference_graph_def,
        input_nodes=input_nodes,
        output_nodes=output_nodes
    )