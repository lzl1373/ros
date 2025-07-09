#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from flask import Flask, Response
import threading

app = Flask(__name__)
# '/home/sunrise/test/test_ros/src/learning_pkg_python/learning_pkg_python/3.mp4'
cap = cv2.VideoCapture(0)  
cv_bridge = CvBridge()

class ImagePublisher(Node):

    def __init__(self, name):
        super().__init__(name)
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        ret, frame = cap.read()
        if ret:
            ros_image = cv_bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.publisher_.publish(ros_image)

def gen_frames():

    while True:
        ret, frame = cap.read() 
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)  
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask():

    app.run(host='0.0.0.0', port=8000, threaded=True)

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher("topic_webcam_pub")

    flask_thread = threading.Thread(target=start_flask)
    flask_thread.daemon = True
    flask_thread.start()

    rclpy.spin(node)

    cap.release()
    node.destroy_node()
    rclpy.shutdown()


