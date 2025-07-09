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
import Hobot.GPIO as GPIO
from typing import List, Tuple, Optional

# Constants
ARM_LENGTHS = (12.0, 12.0, 12.0)  # l1, l2, l3
INITIAL_POSITION = (17.5, 0.0, -5.0)  # x, y, z 
SERIAL_PORT = "/dev/ttyS1"
SERIAL_BAUDRATE = 115200
ULTRASONIC_PINS = (33, 37)  
FLASK_PORT = 8000

VELOCITY = 1000
ACCELERATION = 1
RA_F = 0

vel = 1000
acc = 1
raF = 0
dir = []
clk = []

red_count = 0
green_count = 0
current_time = time.strftime('%H:%M:%S', time.localtime()) 

class SerialSend:
    def __init__(self, port: str = SERIAL_PORT, baudrate: int = SERIAL_BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.connect()

    def connect(self) -> bool:
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            time.sleep(0.1)  
            return True
        except Exception as e:
            print(f"Failed to open serial port {self.port}: {e}")
            self.serial_conn = None
            return False

    def send_data(self, data: List[int]) -> bool:
        if not self.serial_conn and not self.connect():
            return False

        try:
            self.serial_conn.write(bytes(data))
            self.serial_conn.flush()
            return True
        except Exception as e:
            print(f"Serial write failed: {e}")
            self.serial_conn = None
            return False

    def multi_step_pos_ctl(self, directions: List[int], velocity: int, 
                         acceleration: int, clock_cycles: List[int], ra_f: int) -> bool:
        commands = []
        for motor_id, (direction, clk) in enumerate(zip(directions, clock_cycles), start=1):
            cmd = [
                motor_id, 0xFD, direction,
                (velocity >> 8) & 0xFF, velocity & 0xFF,
                acceleration,
                (clk >> 24) & 0xFF, (clk >> 16) & 0xFF,
                (clk >> 8) & 0xFF, clk & 0xFF,
                ra_f, 0x01, 0x6B
            ]
            commands.append(cmd)
        
        for cmd in commands:
            if not self.send_data(cmd):
                return False
        
        return self.send_data([0, 0xFF, 0x66, 0x6B])

    def multi_step_pos_ctl4(self, directions: List[int], velocity: int, 
                          acceleration: int, clock_cycles: List[int], ra_f: int) -> bool:
        commands = []
        for motor_id, (direction, clk) in enumerate(zip(directions, clock_cycles), start=1):
            cmd = [
                motor_id, 0xFD, direction,
                (velocity >> 8) & 0xFF, velocity & 0xFF,
                acceleration,
                (clk >> 24) & 0xFF, (clk >> 16) & 0xFF,
                (clk >> 8) & 0xFF, clk & 0xFF,
                ra_f, 0x01, 0x6B
            ]
            commands.append(cmd)

        for cmd in commands:
            if not self.send_data(cmd):
                return False
        
        return self.send_data([0, 0xFF, 0x66, 0x6B])

class ArmController:
    def __init__(self):
        self.serial = SerialSend()
        self.old_theta = []
        self.ultrasonic = UltrasonicSensor(*ULTRASONIC_PINS)

    def inverse_kinematics(self, x: float, y: float, z: float) -> Tuple[float, float, float]:

        l1, l2, l3 = ARM_LENGTHS
        r = math.sqrt(x**2 + y**2)
        h = z
        
        theta1 = math.atan2(y, x)
        
        wx = x - l3 * math.cos(theta1)
        wy = y - l3 * math.sin(theta1)
        wz = z
        
        r_wrist = math.sqrt(wx**2 + wy**2)
        h_wrist = wz
        
        D = (r_wrist**2 + h_wrist**2 - l1**2 - l2**2) / (2 * l1 * l2)
        
        if abs(D) > 1:
            raise ValueError("Target position out of workspace")
        
        theta3 = math.atan2(math.sqrt(1 - D**2), D)
        
        k1 = l1 + l2 * math.cos(theta3)
        k2 = l2 * math.sin(theta3)
        theta2 = math.atan2(h_wrist, r_wrist) - math.atan2(k2, k1)
        
        return math.degrees(theta1), math.degrees(theta2), math.degrees(theta3)

    def start_arm(self, x: float, y: float, z: float) -> bool:
        try:
            theta1, theta2, theta3 = self.inverse_kinematics(x, y, z)
            self.old_theta = [theta1, theta2, theta3]
            
            theta1 = theta1 
            theta2 = -theta2
            theta3 = theta3 - 90 - theta2
            
            directions = []
            clock_cycles = []
            for angle, factor in [(theta1, 40), (theta2, 40), (theta3, 42)]:
                directions.append(0 if angle < 0 else 1)
                clock_cycles.append(int(factor * abs(angle)))
            
            return self.serial.multi_step_pos_ctl(directions, VELOCITY, ACCELERATION, clock_cycles, RA_F)
            
        except ValueError as e:
            print(f"Kinematics error: {e}")
            return False

    def control_arm(self, x: float, y: float, z: float, is_return: bool = False) -> bool:
        try:
            theta1, theta2, theta3 = self.inverse_kinematics(x, y, z)
            
            if not self.old_theta:
                return self.start_arm(x, y, z)
            
            # Calculate relative movement
            theta1 -= self.old_theta[0]
            theta2 -= self.old_theta[1]
            adjustment = theta2 if theta2 > 0 else -theta2
            theta3 = theta3 - self.old_theta[2] + adjustment
            
            directions = []
            clock_cycles = []
            for angle, factor in [(theta1, 40), (theta2, 40), (theta3, 42)]:
                directions.append(0 if angle < 0 else 1)
                clock_cycles.append(int(factor * abs(angle)))
            
            if is_return:
                directions = [1 - d for d in directions]  
            
            self.old_theta = [
                self.old_theta[0] + theta1,
                self.old_theta[1] + theta2,
                self.old_theta[2] + theta3 - adjustment
            ]
            
            return self.serial.multi_step_pos_ctl(directions, VELOCITY, ACCELERATION, clock_cycles, RA_F)
            
        except ValueError as e:
            print(f"Kinematics error: {e}")
            return False

    def control_arm_go_back(self, x: float, y: float, z: float) -> bool:
        return self.control_arm(x, y, z, is_return=True)

class UltrasonicSensor:
    def __init__(self, trig_pin: int, echo_pin: int):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.trig_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)
        GPIO.output(self.trig_pin, GPIO.LOW)
        time.sleep(0.1)

    def measure_distance(self) -> float:
        GPIO.output(self.trig_pin, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.trig_pin, GPIO.LOW)

        while GPIO.input(self.echo_pin) == GPIO.LOW:
            pulse_start = time.time()

        while GPIO.input(self.echo_pin) == GPIO.HIGH:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # cm
        return round(distance, 2)

    def cleanup(self):
        GPIO.cleanup()

class YoloInferenceNode(Node):
    def __init__(self):
        super().__init__('yolo_inference_node')

        self.image_sub = self.create_subscription(
            Image,
            'image_raw',  
            self.image_callback,
            10
        )
        self.control = ArmController()
        self.serial = SerialSend()
        self.UltrasonicSensor = UltrasonicSensor(*ULTRASONIC_PINS)

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
        self.control.start_arm(10,0,10)
        z = self.UltrasonicSensor.measure_distance()
        dir = [0,1,0,1]
        self.serial.multi_step_pos_ctl4(dir,vel,acc,clk,raF)
        self.control.control_arm(x1,0,y1)
        self.control.control_arm_go_back(x1,z,y1)
        print(serial_data)

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

app = Flask(__name__)

@app.route('/image')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control_action():
    try:
        # 获取请求中的 JSON 数据
        data = request.get_json()
        action = data.get('action')
        serial = SerialSend()
        if action:
            # 根据 action 执行相应的操作
            print(f"收到的操作: {action}")
            response = {"status": "success", "message": f"操作 {action} 已成功处理"}
            if action == "start":
                serial_data = 's'
            if action == "Up":
                serial_data = 'u'
                dir = [1,0,1,0]
            if action == "Down":
                serial_data = 'd'
                dir = [0,1,0,1]
            if action == "left":
                serial_data = 'l'
                dir = [0,0,1,1]
            if action == "Right":
                serial_data = 'r'
                dir = [1,1,0,0]
            serial.multi_step_pos_ctl4(dir,vel,acc,clk,raF)
            print(serial_data)
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