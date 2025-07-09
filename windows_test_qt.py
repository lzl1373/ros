import sys
import cv2
import numpy as np
import onnxruntime as ort
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Camera Feed with YOLOv5 Detection')
        self.resize(640, 640)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #282828;  
            }

            QLabel {
                border: 2px solid #0078D7; 
                border-radius: 10px;  
                background-color: white; 
                padding: 10px; 
            }

            QPushButton {
                background-color: #4CAF50; 
                color: white;
                border: none;  
                padding: 10px 20px;  
                border-radius: 5px;  
            }

            QPushButton:hover {
                background-color: #45a049;  
            }
        """)

        layout = QVBoxLayout()

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        self.button = QPushButton('Start Camera', self)
        self.button.clicked.connect(self.toggle_camera)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.model_path = "508_train.onnx" 
        self.session = ort.InferenceSession(self.model_path) 
        self.class_names = ["red pepper", "green pepper"] 

        self.INPUT_WIDTH = 640  
        self.INPUT_HEIGHT = 640  
        self.SCORE_THRESHOLD = 0.5  
        self.NMS_THRESHOLD = 0.45  
        self.CONFIDENCE_THRESHOLD = 0.45 

    def toggle_camera(self):
        print("Button Clicked, toggling camera.")
        if self.cap is None: 
            self.cap = cv2.VideoCapture('3.mp4')
            if not self.cap.isOpened():
                print("Error: Could not open camera.")
                return
            print("Camera opened successfully.")
            self.button.setText('Stop Camera')  
            self.timer.start(30)  
        else:  
            self.cap.release()
            self.cap = None
            self.button.setText('Start Camera')  
            self.timer.stop() 

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            return

        result_frame = self.run_inference(frame, self.class_names)

        result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        h, w, ch = result_frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(result_frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)

        desired_width = 640  
        desired_height = 640 
        pixmap = pixmap.scaled(desired_width, desired_height, Qt.AspectRatioMode.KeepAspectRatio) 

        self.label.setPixmap(pixmap)

    def preprocess_image(self, image, input_size=(640, 640)):
        image_resized = cv2.resize(image, input_size, interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_normalized = image_rgb.astype(np.float32) / 255.0
        image_input = np.expand_dims(np.transpose(image_normalized, (2, 0, 1)), axis=0)
        return image, image_input

    def post_process(self, input_image, outputs, class_names):
        confidences, boxes = [], []
        print(f"输出形状: {outputs.shape}")

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
            cv2.rectangle(input_image, (left, top), (left + width, top + height), (0, 255, 0), 2)
            #cv2.putText(input_image, f'{class_names[i]}: ', (left, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(input_image, 'red pepper: ', (left, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(input_image, f'{confidences[i]:.2f}', (left + 50, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            coord_text = f"({left},{top})"
            cv2.putText(input_image, coord_text, (left, top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)


        return input_image

    def run_inference(self, image, class_names):

        image, image_input = self.preprocess_image(image)

        outputs = self.session.run(None, {self.session.get_inputs()[0].name: image_input})[0].squeeze(axis=0)

        result_image = self.post_process(image, outputs, class_names)

        return result_image

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
