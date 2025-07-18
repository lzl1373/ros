# 火爆骑士：基于ROS的红绿辣椒采摘分类智能机器人

本项目基于机器人操作系统（ROS）和YOLOv5（ONNX模型），实现了一个用于红绿辣椒采摘的智能机器人。机器人能够实时检测辣椒，并通过串口通信控制机械臂和底盘执行抓取操作。同时，项目还集成了Flask Web服务器，用于实时视频流传输和控制指令的交互。

## 功能特点

- **YOLOv5推理**：使用YOLOv5 ONNX模型进行目标检测，准确识别红绿辣椒。
- **实时视频流传输**：通过Flask Web界面实时显示处理后的视频帧，方便用户监控辣椒采摘过程。
- **串口通信**：将检测到的目标坐标和类别信息通过串口发送给外部设备，实现对机械臂和底盘的控制。
- **动态标签显示**：在视频流中实时显示检测到的目标标签及相关信息，辅助机器人进行抓取决策。
- **控制接口**：支持通过 Web 接口发送控制指令，指令通过 RDK 接收并通过串口发送到机械臂控制系统。

## 硬件要求

- 电机驱动模块（UART控制）
- 机械臂执行机构
- 摄像头（用于图像采集）
- 超声波模块（用于测量物体距离）

## 安装与运行

### 环境要求

- **ROS 2**：Foxy、Galactic 或 Humble 版本
- **Python 3.8及以上**
- **OpenCV**
- **ONNX Runtime**
- **Flask**
- **cv_bridge**
- **pyserial**

### 安装依赖

首先安装项目所需的Python依赖包：

```bash
pip install opencv-python onnxruntime flask pyserial opencv-python requests
```

### ROS 2 环境配置

1. 安装 ROS 2（如果尚未安装）。
2. 创建并初始化 ROS 2 工作空间，并执行环境配置。
3. 将 Python 脚本放置于你的 ROS 包目录中。

### 启动项目

#### 启动 ROS 2 节点

在终端中运行以下命令启动 ROS 2 节点：

```bash
ros2 run <你的包名> <你的脚本名>
```

#### 启动 Flask 服务器

Flask 服务器会自动启动，您可以在浏览器中访问以下地址查看视频流：

```bash
http://<设备IP地址>:8000/image
```

## 文件结构

```bash
.
├── src/learning_pkg_python/
│   ├── learning_pkg_python/
│   │   ├── test*.py    # 主要推理脚本
│   │   └── ...         # 其他源码文件
├── models/
│   ├── *.onnx          # YOLOv5 ONNX 模型文件
├── README.md           # 项目说明文档
└── ...                 # 其他项目文件
```

## 项目说明

### 图像数据获取与处理

1. **摄像头图像获取**：使用 ROS 节点获取摄像头图像并将其通过 `image_raw` 话题发布。
2. **YOLOv5 推理**：`YoloInferenceNode` 类订阅 `image_raw` 话题，接收图像数据并使用 YOLOv5 ONNX模型进行目标检测。识别出红绿辣椒后，提取目标的平面坐标。

### 超声波模块测距

使用超声波模块（`UltrasonicSensor.measure_distance()`）测量目标与机器人之间的水平距离，以帮助机器人在抓取操作时判断合适的位置。

### 逆向运动学计算

使用 `inverse_kinematics` 解算机械臂的角度，并通过control.control()函数控制，以确保机械臂准确到达目标位置并完成抓取操作。

### 串口通信

通过 `serial` 库，将目标的平面坐标、类别信息以及超声波测得的距离数据发送给控制系统。该数据用于驱动底盘和机械臂进行相应的操作。

### Flask Web 服务器

Flask 应用创建 `/image` 路由，实时推送处理后的图像数据到网页上，用户可以通过 Web 界面监控视频流并发送控制指令。

### 控制接口

提供一个 `/control` 接口，用于接收来自 Web 客户端的控制指令。指令通过 RDK 接收，并通过串口发送到机械臂和底盘控制系统执行相应操作。

**控制指令请求示例**：

```json
{
  "action": "start"
}
```

### 串口通信实现

通过串口，发送检测到的目标位置信息、类别及超声波测距结果到外部设备，驱动机械臂和底盘的运动。通信的稳定性和及时性对机器人性能至关重要。

## 硬件部分

### 电机驱动模块

- 用于控制机器人底盘的运动（前进、后退、左移、右移等）。

### 机械臂控制

- 控制机械臂完成红绿辣椒的抓取、放置及其它操作。
