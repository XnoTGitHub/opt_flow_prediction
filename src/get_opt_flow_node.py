#!/usr/bin/env python3
import sys
import cv2
import numpy as np
#import tensorflow as tf
#from tensorflow import keras
from imantics import Mask
import rospy
from thesis.msg import ObjectArray, LabeledPolygon, CloudArray, LabeledCloud
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import Point32, Polygon, Twist
from rospy.numpy_msg import numpy_msg
import rospkg
from dynamic_reconfigure import client
import time

from cv_bridge import CvBridge, CvBridgeError

import torch
import torch.nn as nn
import torch.nn.functional as F

from ResNet import Encoder, Binary, Decoder, Autoencoder, Bottleneck


from PIL import Image as ImagePIL

use_gpu = True


SUBSCRIBING_TOPIC_NAME = 'pred_depth'
SUBSCRIBING_TOPIC_INFO = 'camera/rgb/camera_info'

PUBLISHING_IMAGE_TOPIC_NAME = 'opt_flow/image'
PUBLISHING_EMBEDDING_TOPIC_NAME = 'opt_flow/embedding'


class Opt_Flow_Prediction_Service:

    def __init__(self):
        # comment this line what is running on cpu/gpu
        # tf.debugging.set_log_device_placement(True)
        rospack = rospkg.RosPack()
        base_path = rospack.get_path("opt_flow_prediction")

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        self.zsize = 64

        self.encoder = Encoder(Bottleneck, [3, 4, 6, 3]).to(self.device)
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).to(self.device)
        self.encoder.fc = nn.Linear(65536, self.zsize).to(self.device)
        self.encoder=self.encoder.to(self.device)

        #self.decoder = Decoder().to(self.device)

        self.autoencoder = Autoencoder(self.encoder).to(self.device)

        self.autoencoder.load_state_dict(torch.load(base_path + '/src/ResNet18_Same_Dataset.pt',map_location=torch.device(self.device)))
        self.autoencoder.eval()


        self.publisher = rospy.Publisher(PUBLISHING_IMAGE_TOPIC_NAME, Image, queue_size=1)
        self.publisher_embedding = rospy.Publisher(PUBLISHING_EMBEDDING_TOPIC_NAME, Float32MultiArray, queue_size=1)




    def predict(self, image):

        image = image[np.newaxis]
        image_tensor = torch.tensor(image.copy())
        image_tensor = image_tensor.float()/255.

        self.embedding = self.autoencoder.encoder(image_tensor)
        print(self.embedding)
        opt_flow_image = self.autoencoder.decoder(self.embedding)

        return opt_flow_image




    def opt_flow_predict(self, image):

        # see: https://wiki.ros.org/rospy_tutorials/Tutorials/numpy
        img_array = np.frombuffer(image.data, dtype=np.uint8).reshape(-1, image.width,image.height, order='F')
        print(img_array.shape)
        img_array = np.transpose(img_array, (2, 1, 0 ))
        print('number of channels: ',img_array.shape[2])
        #if img_array.shape[2] == 4:
        #    img_array = img_array[:, :, :-1]
        #img_array = img_array[:, :, ::-1]
        img_array = np.transpose(img_array, (2, 0, 1 ))
        print(img_array.shape)
        print(img_array.max())
        output = self.predict(img_array)
        print('output.shape: ',output.shape)

        embedding_output = self.embedding.cpu().detach().numpy()
        print(embedding_output.shape)
        embedding_output_frame = Float32MultiArray()
        embedding_output_frame.data = embedding_output[0,:].tolist()
        print(type(embedding_output_frame.data[0]))



        self.publisher_embedding.publish(embedding_output_frame)

        #msg = Twist()
        #msg.linear.x = 0.2
        #msg.angular.z = output[0][1]
        np_arr = output[0].cpu().detach().numpy()
        np_arr = np.transpose(np_arr, (1, 2, 0 ))
        np_arr = 255 * np_arr
        np_arr = np_arr.astype(np.uint8)


        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.height = 192
        msg.width = 640
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.data = np_arr.tobytes()
        self.publisher.publish(msg)

        print(np_arr.shape)
        ##############################



def opt_flow_prediction():
    rospy.init_node("Depth_Prediction", anonymous=True)
    # fetch a single message from camera_info
    #rospy.loginfo("Waiting on camera info")
    #camera_info = rospy.wait_for_message(SUBSCRIBING_TOPIC_INFO, CameraInfo)
    #rospy.loginfo("done")
    #rospy.loginfo(camera_info)

    rospy.loginfo('Start predicting Optical Flow from Depth Image')
    opt_flow_node = Opt_Flow_Prediction_Service()
    rospy.Subscriber(SUBSCRIBING_TOPIC_NAME, numpy_msg(Image), opt_flow_node.opt_flow_predict,queue_size=1)

    rospy.spin()


if __name__ == '__main__':
    try:
        print("Node started")
        opt_flow_prediction()
    except rospy.ROSInterruptException:
        pass
