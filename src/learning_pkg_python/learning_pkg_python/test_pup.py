#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy                        
from rclpy.node import Node         
from sensor_msgs.msg import Image  
from cv_bridge import CvBridge      
import cv2                         

class ImagePublisher(Node):

    def __init__(self, name):
        super().__init__(name)                                           
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)  
        self.timer = self.create_timer(0.1, self.timer_callback)         
        #'/home/sunrise/test/test_ros/src/learning_pkg_python/learning_pkg_python/3.mp4'
        self.cap = cv2.VideoCapture(0)                                  
        self.cv_bridge = CvBridge()                                      

    def timer_callback(self):
        ret, frame = self.cap.read()                                     
        
        if ret == True:                                                  
            self.publisher_.publish(
                self.cv_bridge.cv2_to_imgmsg(frame, 'bgr8'))             

        self.get_logger().info('Publishing video frame')                 

def main(args=None):                                 
    rclpy.init(args=args)                          
    node = ImagePublisher("topic_webcam_pub")       
    rclpy.spin(node)                              
    node.destroy_node()                             
    rclpy.shutdown()                               
