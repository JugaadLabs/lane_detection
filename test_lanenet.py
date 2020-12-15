#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image
"""
import argparse
import os.path as ops
import os
import time
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')



def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_path', type=str, help='The source tusimple lane test data dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--save_dir', type=str, help='The test output save root dir')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet(image_path, weights_path):
    """

    :param image_path:
    :param weights_path:
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)
    img_names = os.listdir(base_img_dir)
    img_paths = [base_img_dir+i for i in img_names]
    net = lanenet.LaneNet(phase='test', cfg=CFG)
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')
    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)
    for image_path in img_paths:
        print(image_path)
        LOG.info('Start reading image and preprocessing')
        t_start = time.time()
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_vis = image
        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0
        LOG.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))



        # Set sess configuration
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'

        sess = tf.Session(config=sess_config)

        # define moving average version of the learned variables for eval
        with tf.variable_scope(name_or_scope='moving_avg'):
            variable_averages = tf.train.ExponentialMovingAverage(
                CFG.SOLVER.MOVING_AVE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()

        # define saver
        saver = tf.train.Saver(variables_to_restore)

        with sess.as_default():
            saver.restore(sess=sess, save_path=weights_path)

            t_start = time.time()
            loop_times = 1
            for i in range(loop_times):
                binary_seg_image, instance_seg_image = sess.run(
                    [binary_seg_ret, instance_seg_ret],
                    feed_dict={input_tensor: [image]}
                )
            
            t_cost = time.time() - t_start
            t_cost /= loop_times
            LOG.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))

            postprocess_result = postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis
            )
            mask_image = postprocess_result['mask_image']

            for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
                instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
            embedding_image = np.array(instance_seg_image[0], np.uint8)
            
            plt.figure('mask_image')
            plt.imshow(mask_image[:, :, (2, 1, 0)])
            plt.figure('src_image')
            plt.imshow(image_vis[:, :, (2, 1, 0)])
            plt.figure('instance_image')
            plt.imshow(embedding_image[:, :, (2, 1, 0)])
            plt.figure('binary_image')
            plt.imshow(binary_seg_image[0] * 255, cmap='gray')
            plt.show()

    sess.close()

    return

def custom_test_lanenet(src_dir , weights_path, save_dir):
    """

    :param src_dir:
    :param weights_path:
    :param save_dir:
    :return:
    """
    assert ops.exists(src_dir), '{:s} not exist'.format(src_dir)

    os.makedirs(save_dir, exist_ok=True)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():
        save_img_lane_dict = {}
        saver.restore(sess=sess, save_path=weights_path)
        img_names = os.listdir(src_dir)

        image_list = [os.path.join(src_dir,i) for i in img_names]
        avg_time_cost = []
        for index, image_path in tqdm.tqdm(enumerate(image_list), total=len(image_list)):

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_vis = image
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0

            t_start = time.time()
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )
            avg_time_cost.append(time.time() - t_start)

            postprocess_result = postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis
            )
            lanes = postprocess_result['lane_points_on_img']
            
            if index % 100 == 0:
                LOG.info('Mean inference time every single image: {:.5f}s'.format(np.mean(avg_time_cost)))
                avg_time_cost.clear()

            input_image_dir = src_dir
            input_image_name = ops.split(image_path)[-1]
            output_image_dir = save_dir
            os.makedirs(output_image_dir, exist_ok=True)
            output_image_path = ops.join(output_image_dir, input_image_name)
            # if ops.exists(output_image_path):
            #     continue
            save_img_lane_dict[input_image_name] = lanes
            cv2.imwrite(output_image_path, postprocess_result['source_image'])
    output_json_name = os.path.join(save_dir,'img_lane_points.json')
    with open(output_json_name, 'w') as fp:
        json.dump(save_img_lane_dict, fp)
    return

def custom_test_lanenet_video(src_dir , weights_path, save_dir):
    """

    :param src_dir:
    :param weights_path:
    :param save_dir:
    :return:
    """
    assert ops.exists(src_dir), '{:s} not exist'.format(src_dir)

    os.makedirs(save_dir, exist_ok=True)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('output.mp4', fourcc, 4.0, (640,480))
    out = cv2.VideoWriter(os.path.join(save_dir,'output.mp4'),0x7634706d, 4, (1280,720))
    with sess.as_default():
        save_img_lane_dict = {}
        saver.restore(sess=sess, save_path=weights_path)
        cap = cv2.VideoCapture(src_dir)

        avg_time_cost = []
        frame_id = 0
        while(True):
            ret, image = cap.read()
            frame_id+=1
            if not ret or frame_id>50:
                break
            image_vis = image
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0

            t_start = time.time()
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )
            avg_time_cost.append(time.time() - t_start)

            postprocess_result = postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis
            )
            lanes = postprocess_result['lane_points_on_img']
            
            if frame_id % 100 == 0:
                LOG.info('Mean inference time every single image: {:.5f}s'.format(np.mean(avg_time_cost)))
                avg_time_cost.clear()

            output_image_dir = save_dir
            os.makedirs(output_image_dir, exist_ok=True)
            output_image_path = ops.join(output_image_dir, str(frame_id)+'.jpg')
            # if ops.exists(output_image_path):
            #     continue
            save_img_lane_dict[frame_id] = lanes
            cv2.imwrite(output_image_path, postprocess_result['source_image'])
            out.write(postprocess_result['source_image'])
    out.release()
    cap.release()
    output_json_name = os.path.join(save_dir,'img_lane_points.json')
    with open(output_json_name, 'w') as fp:
        json.dump(save_img_lane_dict, fp)
    return

if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()
    # custom_test_lanenet(
    #     src_dir=args.image_dir,
    #     weights_path=args.weights_path,
    #     save_dir=args.save_dir
    # )
    custom_test_lanenet_video(
        src_dir=args.vid_path,
        weights_path=args.weights_path,
        save_dir=args.save_dir
    )
