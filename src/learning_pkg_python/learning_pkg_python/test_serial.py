#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import onnxruntime as ort
import cv2
import numpy as np
import serial
from flask import Flask, request, jsonify, Response
import threading
import time 
import math
import sys
import signal
import Hobot.GPIO as GPIO

TRIG = 33
ECHO = 37

l1 = 12.0  
l2 = 12.0 
l3 = 12.0 

index = 0
vel = 1000
acc = 1
raF = 0
dir = []
clk = []

err_x = 0.0
err_y = 0.0
err_z = 0.0
x = 17.5  + err_x
y = 0.0 + err_y
z = -5 + err_z

old_theta = []
new = []

ser = serial.Serial('/dev/ttyS1', 115200, timeout=1)
app = Flask(__name__)

result_image = None

labels = [("Label1", (50, 50)), ("Label2", (150, 150)), ("Label3", (250, 250))] 

red_count = 0
green_count = 0
current_time = time.strftime('%H:%M:%S', time.localtime()) 

def signal_handler(signal, frame):
    sys.exit(0)

GPIO.setwarnings(False)

def measure_distance():
    GPIO.output(TRIG, GPIO.LOW)
    time.sleep(0.1)

    GPIO.output(TRIG, GPIO.HIGH)
    time.sleep(0.00001)  
    GPIO.output(TRIG, GPIO.LOW)

    while GPIO.input(ECHO) == GPIO.LOW:
        pulse_start = time.time()

    while GPIO.input(ECHO) == GPIO.HIGH:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start

    distance = pulse_duration * 17150  

    return round(distance, 2)

def dis():
    GPIO.setmode(GPIO.BOARD)  
    GPIO.setup(TRIG, GPIO.OUT) 
    GPIO.setup(ECHO, GPIO.IN)
    distance = 0
    try:
        distance = measure_distance()  
        print(f"Distance: {distance} cm")
        time.sleep(1)

    except KeyboardInterrupt:
        print("程序被用户中断")
        GPIO.cleanup()  

    return distance

def inverse_kinematics_3d(x, y, z, l1, l2, l3):

    r = np.sqrt(x**2 + y**2)

    h = z
    
    theta1 = np.arctan2(y, x)
    

    wx = x - l3 * np.cos(theta1)
    wy = y - l3 * np.sin(theta1)
    wz = z
    

    r_wrist = np.sqrt(wx**2 + wy**2)
    h_wrist = wz
    

    D = (r_wrist**2 + h_wrist**2 - l1**2 - l2**2) / (2 * l1 * l2)
    '''
    C = (a**2 + b**2 - c**2) / 2*a*b
    '''

    if abs(D) > 1:
        raise ValueError("目标位置超出机械臂工作空间")
    
    theta3 = np.arctan2(np.sqrt(1 - D**2), D) 
    
    k1 = l1 + l2 * np.cos(theta3)
    k2 = l2 * np.sin(theta3)
    theta2 = np.arctan2(h_wrist, r_wrist) - np.arctan2(k2, k1)
     
    return np.degrees(theta1), np.degrees(theta2), np.degrees(theta3)


def serial1_send(data):
    try:
        ser = serial.Serial("/dev/ttyS1",115200, timeout=1) 
        test_data = data
        #print(test_data)
        ser.write(test_data) 
        ser.close()  
    except Exception as e:
        print("打开或写入串口 '/dev/ttyS1' 失败:", e)

def multi_step_pos_ctl(dir,vel,acc,clk,raF):
    cmd1 = [0x01,0xFD,dir[0],(vel>>8)&0xFF,vel&0xFF,acc,(clk[0] >> 24) & 0xFF,(clk[0] >> 16) & 0xFF,(clk[0] >> 8) & 0xFF,clk[0] & 0xFF,raF,0x01,0x6B]
    cmd2 = [0x02,0xFD,dir[1],(vel>>8)&0xFF,vel&0xFF,acc,(clk[1] >> 24) & 0xFF,(clk[1] >> 16) & 0xFF,(clk[1] >> 8) & 0xFF,clk[1] & 0xFF,raF,0x01,0x6B]
    cmd3 = [0x03,0xFD,dir[2],(vel>>8)&0xFF,vel&0xFF,acc,(clk[2] >> 24) & 0xFF,(clk[2] >> 16) & 0xFF,(clk[2] >> 8) & 0xFF,clk[2] & 0xFF,raF,0x01,0x6B]
    serial1_send(cmd1)
    serial1_send(cmd2)
    serial1_send(cmd3)    
    cmd = [0,0xFF,0x66,0x6B]
    serial1_send(cmd)

def multi_step_pos_ctl4(dir,vel,acc,clk,raF):
    cmd1 = [0x01,0xFD,dir[0],(vel>>8)&0xFF,vel&0xFF,acc,(clk >> 24) & 0xFF,(clk >> 16) & 0xFF,(clk >> 8) & 0xFF,clk[0] & 0xFF,raF,0x01,0x6B]
    cmd2 = [0x02,0xFD,dir[1],(vel>>8)&0xFF,vel&0xFF,acc,(clk >> 24) & 0xFF,(clk >> 16) & 0xFF,(clk >> 8) & 0xFF,clk[1] & 0xFF,raF,0x01,0x6B]
    cmd3 = [0x03,0xFD,dir[2],(vel>>8)&0xFF,vel&0xFF,acc,(clk >> 24) & 0xFF,(clk >> 16) & 0xFF,(clk >> 8) & 0xFF,clk[2] & 0xFF,raF,0x01,0x6B]
    cmd4 = [0x03,0xFD,dir[3],(vel>>8)&0xFF,vel&0xFF,acc,(clk >> 24) & 0xFF,(clk >> 16) & 0xFF,(clk >> 8) & 0xFF,clk[3] & 0xFF,raF,0x01,0x6B]
    serial1_send(cmd1)
    serial1_send(cmd2)
    serial1_send(cmd3)    
    serial1_send(cmd4)
    cmd = [0,0xFF,0x66,0x6B]
    serial1_send(cmd)

def append_direction(value, dir):
    if value > 0:
        dir.append(1)
    else:
        dir.append(0)    

def append_clk(clk,theta1,theta2,theta3):
    #theta1
    value1 = 40 * abs(theta1)
    clk.append(int(value1))
    #theta2
    value2 = 40 * abs(theta2)
    clk.append(int(value2))
    #theta3 
    value3 = 42 * abs(theta3)  - value2 / 42 
    clk.append(int(value3))

def start_arm():
    try:
        theta1, theta2, theta3 = inverse_kinematics_3d(x, y, z, l1, l2, l3)
        old_theta.append(theta1)
        old_theta.append(theta2)
        old_theta.append(theta3)
        #print(f"start_old  关节角度: θ1 = {theta1:.2f}°, θ2 = {theta2:.2f}°, θ3 = {theta3:.2f}°")
        theta1 = theta1 
        theta2 = - theta2
        theta3 = theta3 - 90 - theta2  
        #print(f"start_new  关节角度: θ1 = {theta1:.2f}°, θ2 = {theta2:.2f}°, θ3 = {theta3:.2f}°")
        append_direction(theta1, dir)
        append_direction(theta2, dir)
        append_direction(theta3, dir)
        append_clk(clk,theta1,theta2,theta3)
        print('dir:',dir)
        print('clk:',clk)
        #serial_send.multi_step_pos_ctl(dir,vel,acc,clk,raF)   
    except ValueError as e:
        print(f"错误: {e}")
     
def control_arm(x,y,z):
    try:
        theta1, theta2, theta3 = inverse_kinematics_3d(x, y, z, l1, l2, l3)
        #print(f"old  关节角度: θ1 = {theta1:.2f}°, θ2 = {theta2:.2f}°, θ3 = {theta3:.2f}°")
        theta1 = theta1 - old_theta[0]
        theta2 = theta2 - old_theta[1]
        if theta2 > 0:
            theta3 = theta3 - old_theta[2] + theta2
        else:  
            theta3 = theta3 - old_theta[2] - theta2   
        #old_theta.clear()  
        #print(f"new  关节角度: θ1 = {theta1:.2f}°, θ2 = {theta2:.2f}°, θ3 = {theta3:.2f}°")
        append_direction(theta1, dir)
        append_direction(-theta2, dir)
        append_direction(theta3, dir)
        append_clk(clk,theta1,theta2,theta3)
        print('dir:',dir)
        print('clk:',clk)
        multi_step_pos_ctl(dir,vel,acc,clk,raF)
    
    except ValueError as e:
        print(f"错误: {e}")

def control_arm_go_back(x,y,z):
    try:
        theta1, theta2, theta3 = inverse_kinematics_3d(x, y, z, l1, l2, l3)
        #print(f"old  关节角度: θ1 = {theta1:.2f}°, θ2 = {theta2:.2f}°, θ3 = {theta3:.2f}°")
        theta1 = theta1 - old_theta[0]
        theta2 = theta2 - old_theta[1]
        if theta2 > 0:
            theta3 = theta3 - old_theta[2] + theta2
        else:  
            theta3 = theta3 - old_theta[2] - theta2       
        #print(f"new  关节角度: θ1 = {theta1:.2f}°, θ2 = {theta2:.2f}°, θ3 = {theta3:.2f}°")
        append_direction(-theta1, dir)
        append_direction(theta2, dir)
        append_direction(-theta3, dir)
        append_clk(clk,theta1,theta2,theta3)
        print('dir:',dir)
        print('clk:',clk)
        multi_step_pos_ctl(dir,vel,acc,clk,raF)
    
    except ValueError as e:
        print(f"错误: {e}")       

def control(x,y,z):
    global index 
    start_x = 259 + 10
    start_y = 240 - 16#(30像素 1cm) #y-为向左移  x-为向内移
    if index != 0:
        dir = [0,1,0,1]
        multi_step_pos_ctl4(dir,vel,acc,int(y*3200/24.5),raF)
        center_x,center_y = x,z
        if center_y * center_x:
            print('center_X:', center_x, ', center_Y:', center_y)  
            if center_x < 300:
                center_x -= 10
            elif center_x > 300:
                center_x += 15 
            new_x = x + (start_x - center_x) / 30.5
            new_y = y + (start_y - center_y) / 30.5
            print('center_new_X:', new_x, ', center_new_Y:', new_y)
            clk.clear()
            dir.clear()
            control_arm(new_x,new_y,10.9)
            clk.clear()
            dir.clear()
            control_arm_go_back(new_x,new_y,10.9)

class YoloInferenceNode(Node):
    def __init__(self):
        super().__init__('yolo_inference_node')

        self.image_sub = self.create_subscription(
            Image,
            'image_raw',  
            self.image_callback,
            10
        )

        self.bridge = CvBridge()

        self.model_path = "/home/sunrise/test/test_ros/src/learning_pkg_python/learning_pkg_python/508_train.onnx"  
        self.session = ort.InferenceSession(self.model_path)  
        self.class_names = ["red pepper", "green pepper"] 

        self.INPUT_WIDTH = 640  
        self.INPUT_HEIGHT = 640  
        self.SCORE_THRESHOLD = 0.5  
        self.NMS_THRESHOLD = 0.45 
        self.CONFIDENCE_THRESHOLD = 0.45  

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.get_logger().info(f"Sending data to serial 1")
        
        session = self.load_model(self.model_path)

        image, image_input = self.preprocess_image(cv_image)
        
        result = session.run(None, {session.get_inputs()[0].name: image_input})[0].squeeze(axis=0)

        global result_image

        result_image = self.post_process(cv_image, result, self.class_names, 1)

        return result_image
    
    def load_model(self, model_path):
        session = ort.InferenceSession(model_path)
        return session

    def preprocess_image(self, image, input_size=(640, 640)):
        image_resized = cv2.resize(image, input_size, interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_normalized = image_rgb.astype(np.float32) / 255.0
        image_input = np.expand_dims(np.transpose(image_normalized, (2, 0, 1)), axis=0)
        return image, image_input

    def post_process(self, input_image, outputs, class_names, flag):
        global egg_num
        confidences, boxes = [], []
        rows = outputs.shape[1] if len(outputs.shape) == 3 else outputs.shape[0]
        image_height, image_width = input_image.shape[:2]
        x_factor, y_factor = image_width / self.INPUT_WIDTH, image_height / self.INPUT_HEIGHT

        for r in range(rows):
            row = outputs[0][r] if len(outputs.shape) == 3 else outputs[r]
            confidence = row[4]
            if confidence >= self.CONFIDENCE_THRESHOLD:
                class_scores = row[5:]
                class_id = np.argmax(class_scores)
                if class_scores[class_id] > self.SCORE_THRESHOLD:
                    confidences.append(confidence)
                    cx, cy, w, h = row[:4]
                    left = int((cx - w / 2) * x_factor)
                    top = int((cy - h / 2) * y_factor)
                    width, height = int(w * x_factor), int(h * y_factor)
                    boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        if len(indices) == 0:   
            return input_image    
            
        for i in indices.flatten():
            left, top, width, height = boxes[i]
            data_x = left + width / 2  
            data_y = top + height / 2
            self.send_serial_data(data_x, data_y, class_names[0])
            if flag:
                global red_count, green_count
                if class_names[0] == "red pepper":
                    red_count += 1
                if class_names[0] == "green pepper":
                    green_count += 1    
                cv2.rectangle(input_image, (left, top), (left + width, top + height), (0, 255, 0), 2)
                cv2.putText(input_image, class_names[0], (left, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(input_image, f'{confidences[i]:.2f}', (left + 50, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                coord_text = f"({left},{top})"
                cv2.putText(input_image, coord_text, (left, top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        return input_image

    def send_serial_data(self, x1, y1, cls):
        serial_data = f"{x1},{y1},{cls}\n" 
        start_arm()
        z = dis()
        control(x1,z,y1)
        print(serial_data)
        #ser.write(serial_data.encode())

def gen_frames():
    global result_image
    while True:
        if result_image is not None:
            frame = result_image.copy()
            label_text = f"red: {red_count}  green: {green_count}  time: {current_time}"
            cv2.putText(frame, label_text, (frame.shape[1] - 600, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/image')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control_action():
    try:
        # 获取请求中的 JSON 数据
        data = request.get_json()
        action = data.get('action')

        if action:
            # 根据 action 执行相应的操作
            print(f"收到的操作: {action}")
            response = {"status": "success", "message": f"操作 {action} 已成功处理"}
            if action == "start":
                serial_data = 's'
            if action == "Up":
                serial_data = 'u'
            if action == "Down":
                serial_data = 'd'
            if action == "left":
                serial_data = 'l'
            if action == "Right":
                serial_data = 'r'
            ser.write(serial_data.encode())
        else:
            response = {"status": "error", "message": "未提供有效的操作参数"}

    except Exception as e:
        response = {"status": "error", "message": f"发生错误: {str(e)}"}

    # 返回 JSON 响应
    return jsonify(response)

def start_flask():
    app.run(host='0.0.0.0', port=8000, threaded=True)

def main(args=None):
    rclpy.init(args=args)

    node = YoloInferenceNode()

    flask_thread = threading.Thread(target=start_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
