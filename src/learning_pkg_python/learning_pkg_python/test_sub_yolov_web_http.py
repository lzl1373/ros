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

ser = serial.Serial('/dev/ttyS1', 115200, timeout=1)
app = Flask(__name__)

result_image = None

labels = [("Label1", (50, 50)), ("Label2", (150, 150)), ("Label3", (250, 250))] 

red_count = 0
green_count = 0
current_time = time.strftime('%H:%M:%S', time.localtime()) 

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
                cv2.putText(input_image, 'red:', (left, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(input_image, f'{confidences[i]:.2f}', (left + 50, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                coord_text = f"({left},{top})"
                cv2.putText(input_image, coord_text, (left, top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        return input_image

    def send_serial_data(self, x1, y1, cls):
        serial_data = f"{x1},{y1},{cls}\n" 
        ser.write(serial_data.encode())

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
