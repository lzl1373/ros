# Hot Knight: ROS-based Intelligent Robot for Red and Green Chili Picking and Classification

This project is based on the Robot Operating System (ROS) and YOLOv5 (ONNX model), designed for an intelligent robot that picks red and green chilies. The robot can detect chilies in real-time, control the robotic arm and chassis through serial communication, and display video streams on a web interface. The project integrates a Flask web server for real-time video transmission and control interaction.

## Features

- **YOLOv5 Inference**: Uses YOLOv5 ONNX model for object detection, accurately identifying red and green chilies.
- **Real-time Video Stream**: Displays processed video frames on a Flask web interface for monitoring the chili picking process.
- **Serial Communication**: Sends detected target coordinates and class information through serial communication to control the robotic arm and chassis.
- **Dynamic Label Display**: Shows detected object labels and related information in the video stream, assisting the robot in making grabbing decisions.
- **Control Interface**: Supports sending control commands via a web interface, which are received by the RDK and forwarded through serial communication to the robotic arm control system.

## Hardware Requirements

- Motor driver module (UART control)
- Robotic arm execution mechanism
- Camera (for image capture)
- Ultrasonic module (for measuring object distance)

## Installation and Running

### Environment Requirements

- **ROS 2**: Foxy, Galactic, or Humble versions
- **Python 3.8 or higher**
- **OpenCV**
- **ONNX Runtime**
- **Flask**
- **cv_bridge**
- **pyserial**

### Install Dependencies

First, install the required Python dependencies:

```bash
pip install opencv-python onnxruntime flask pyserial opencv-python requests
```

### ROS 2 Environment Configuration

1. Install ROS 2 (if not installed).
2. Create and initialize a ROS 2 workspace, and configure the environment.
3. Place the Python scripts in your ROS package directory.

### Start the Project

#### Start ROS 2 Nodes

Run the following command in the terminal to start the ROS 2 node:

```bash
ros2 run <your_package_name> <your_script_name>
```

#### Start Flask Server

The Flask server will start automatically, and you can access the video stream in the browser at the following address:

```bash
http://<device_IP_address>:8000/image
```

## File Structure

```bash
.
├── src/learning_pkg_python/
│   ├── learning_pkg_python/
│   │   ├── test*.py    # Main inference script
│   │   └── ...         # Other source files
├── models/
│   ├── *.onnx          # YOLOv5 ONNX model files
├── README.md           # Project documentation
└── ...                 # Other project files
```

## Project Description

### Image Data Acquisition and Processing

1. **Camera Image Acquisition**: Use ROS nodes to obtain camera images and publish them via the `image_raw` topic.
2. **YOLOv5 Inference**: The `YoloInferenceNode` class subscribes to the `image_raw` topic, receives image data, and uses the YOLOv5 ONNX model for object detection. After identifying red and green chilies, it extracts the planar coordinates of the targets.

### Ultrasonic Module Distance Measurement

The ultrasonic module (`UltrasonicSensor.measure_distance()`) is used to measure the horizontal distance between the robot and the target, helping the robot determine the optimal position for grabbing.

### Inverse Kinematics Calculation

The inverse kinematics algorithm (`inverse_kinematics`) calculates the angles for the robotic arm, which are then used to accurately reach the target location and complete the grabbing operation.

### Serial Communication

The target coordinates, class information, and distance data from the ultrasonic sensor are sent to the control system via the `serial` library. This data is used to drive the chassis and robotic arm to perform the corresponding operations.

### Flask Web Server

The Flask application creates a `/image` route to push processed image data to the web interface. Users can monitor the video stream and send control commands via the web interface.

### Control Interface

A `/control` interface is provided to receive control commands from the web client. The commands are received by the RDK and transmitted through serial communication to control the robotic arm and chassis.

**Control Command Request Example**:

```json
{
  "action": "start"
}
```

### Serial Communication Implementation

The serial communication sends the target position information, class data, and ultrasonic distance measurement to external devices, driving the movement of the robotic arm and chassis. The stability and timeliness of communication are crucial to the robot's performance.

## Hardware Components

### Motor Driver Module

- Controls the movement of the robot chassis (forward, backward, left, right, etc.).

### Robotic Arm Control

- Controls the robotic arm to perform tasks like grabbing, placing, and other operations for the red and green chilies.
