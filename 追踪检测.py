# Python版本要求 3.8-3.11
# 需安装部分依赖库，并更新Numpy到最新
#           _____                    _____                    _____                    _____            _____           _______         
#          /\    \                  /\    \                  /\    \                  /\    \          /\    \         /::\    \        
#         /::\    \                /::\    \                /::\    \                /::\____\        /::\____\       /::::\    \       
#        /::::\    \               \:::\    \              /::::\    \              /:::/    /       /:::/    /      /::::::\    \      
#       /::::::\    \               \:::\    \            /::::::\    \            /:::/    /       /:::/    /      /::::::::\    \     
#      /:::/\:::\    \               \:::\    \          /:::/\:::\    \          /:::/    /       /:::/    /      /:::/~~\:::\    \    
#     /:::/  \:::\    \               \:::\    \        /:::/__\:::\    \        /:::/    /       /:::/    /      /:::/    \:::\    \   
#    /:::/    \:::\    \              /::::\    \      /::::\   \:::\    \      /:::/    /       /:::/    /      /:::/    / \:::\    \  
#   /:::/    / \:::\    \    ____    /::::::\    \    /::::::\   \:::\    \    /:::/    /       /:::/    /      /:::/____/   \:::\____\ 
#  /:::/    /   \:::\    \  /\   \  /:::/\:::\    \  /:::/\:::\   \:::\    \  /:::/    /       /:::/    /      |:::|    |     |:::|    |
# /:::/____/     \:::\____\/::\   \/:::/  \:::\____\/:::/  \:::\   \:::\____\/:::/____/       /:::/____/       |:::|____|     |:::|    |
# \:::\    \      \::/    /\:::\  /:::/    \::/    /\::/    \:::\  /:::/    /\:::\    \       \:::\    \        \:::\    \   /:::/    / 
#  \:::\    \      \/____/  \:::\/:::/    / \/____/  \/____/ \:::\/:::/    /  \:::\    \       \:::\    \        \:::\    \ /:::/    /  
#   \:::\    \               \::::::/    /                    \::::::/    /    \:::\    \       \:::\    \        \:::\    /:::/    /   
#    \:::\    \               \::::/____/                      \::::/    /      \:::\    \       \:::\    \        \:::\__/:::/    /    
#     \:::\    \               \:::\    \                      /:::/    /        \:::\    \       \:::\    \        \::::::::/    /     
#      \:::\    \               \:::\    \                    /:::/    /          \:::\    \       \:::\    \        \::::::/    /      
#       \:::\    \               \:::\    \                  /:::/    /            \:::\    \       \:::\    \        \::::/    /       
#        \:::\____\               \:::\____\                /:::/    /              \:::\____\       \:::\____\        \::/____/        
#         \::/    /                \::/    /                \::/    /                \::/    /        \::/    /         ~~              
#          \/____/                  \/____/                  \/____/                  \/____/          \/____/                          
# By AlexWhite 老天保佑封装不翻车 接各类语言开发，前端UI定做，详细联系alexwhite@eschz.eu 

'''
# ---------- 日志抑制，测试无误后可打开减少不必要输出 ---------- #
import os
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
# -------------------------------------------------------------- #
'''        

import ctypes
import cv2
import mediapipe as mp
import pyautogui
import threading
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import time
from queue import Queue

class BlinkControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ciallo～(∠・ω< )⌒★ | 0721.guru | AlexWhite")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 控制参数
        self.running = False
        self.cap = None
        self.ear_threshold = 0.21  # 初始阈值
        self.eye_state = False      
        
        # 状态参数
        self.consecutive_closed = 0  # 连续闭眼帧计数
        self.consecutive_open = 0    # 连续睁眼帧计数
        self.activation_frames = 1   # 触发动作所需闭眼帧
        self.cooldown_frames = 1     # 状态重置所需睁眼帧
        
        # MediaPipe
        self.mp_face = mp.solutions.face_mesh
        self.face = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            static_image_mode=False
        )
        
        # 眼部关键点索引（亚洲人）
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

        # 图像队列
        self.image_queue = Queue(maxsize=3)
        
        # GUI初始化
        self.create_widgets()
        self.start_camera()

        # 头部姿态参数
        self.setup_head_pose()
        
        # 锁定状态参数
        self.last_lock_time = 0
        self.lock_cooldown = 5  # 锁定冷却时间（秒）
        self.nod_counter = 0
        self.nod_required_frames = 10  # 持续帧数阈值，只高不低不然误判！
    def setup_head_pose(self):
        """初始化头部姿态检测参数"""
        # 3D面部参考点（基于亚洲人）
        self.model_points = np.array([
            (0.0, 0.0, 0.0),         # 鼻尖（1号关键点）
            (0.0, -330.0, -65.0),    # 下巴（152号）
            (-225.0, 170.0, -135.0), # 左眼左角（33号）
            (225.0, 170.0, -135.0),  # 右眼右角（263号）
            (-150.0, -150.0, -125.0),# 左嘴角（61号）
            (150.0, -150.0, -125.0)  # 右嘴角（291号）
        ], dtype=np.float64)
        
        # 相机参数（以实际调整）
        self.img_width, self.img_height = 640, 480
        self.focal_length = self.img_width * 1.5
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.img_width/2],
            [0, self.focal_length, self.img_height/2],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        # 头部检测参数
        self.pitch_threshold = 25    
        self.roll_threshold = 30     # 防误触转头阈值
        self.current_pitch = 0    

    def create_widgets(self):
        """创建界面组件"""
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)

        # 视频显示区域
        video_frame = tk.Frame(main_frame)
        video_frame.pack()
        
        self.original_label = tk.Label(video_frame)
        self.original_label.pack(side=tk.LEFT, padx=5)
        
        self.processed_label = tk.Label(video_frame)
        self.processed_label.pack(side=tk.LEFT, padx=5)

        # 控制面板
        control_frame = tk.Frame(main_frame)
        control_frame.pack(pady=10)
        
        self.btn_switch = tk.Button(
            control_frame,
            text="启动检测",
            command=self.toggle_detection,
            width=12,
            height=1
        )
        self.btn_switch.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(
            control_frame,
            text="状态: 未激活",
            fg="gray"
        )
        self.status_label.pack(side=tk.LEFT, padx=5)

        # 设置面板
        debug_frame = tk.Frame(main_frame)
        debug_frame.pack(pady=5)
        
        self.ear_label = tk.Label(
            debug_frame,
            text="EAR:阈值: 0.21",
            font=('Arial', 9)
        )
        self.ear_label.pack(side=tk.LEFT, padx=5)
        
        self.threshold_scale = tk.Scale(
            debug_frame,
            from_=0.15, to=0.99,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            label="灵敏度调节",
            length=200,
            command=self.update_threshold
        )
        self.threshold_scale.set(self.ear_threshold)
        self.threshold_scale.pack(side=tk.LEFT)

    def start_camera(self):
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.update_camera()

    def update_camera(self):
        """更新摄像头画面"""
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.image_queue.qsize() < 3:
                    self.image_queue.put(frame)
                
                # 显示原始画面
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.original_label.imgtk = imgtk
                self.original_label.configure(image=imgtk)

        self.root.after(30, self.update_camera)

    def process_frame(self):
        """核心处理逻辑"""
        while True:
            if not self.image_queue.empty() and self.running:
                frame = self.image_queue.get()
                processed_frame = frame.copy()
                
                try:
                    # 人脸检测
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.face.process(rgb_frame)
                    
                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0].landmark  
                        # 计算EAR值
                        left_ear = self.calculate_ear(landmarks, self.LEFT_EYE_INDICES)
                        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE_INDICES)
                        avg_ear = (left_ear + right_ear) / 2.0
                        
                        # 更新状态
                        if avg_ear < self.ear_threshold:
                            self.consecutive_closed += 1
                            self.consecutive_open = 0
                        else:
                            self.consecutive_open += 1
                            self.consecutive_closed = 0
                        
                        # 触发条件判断
                        if self.consecutive_closed >= self.activation_frames and not self.eye_state:
                            self.eye_state = True
                            current_x, current_y = pyautogui.position()
                            pyautogui.click(current_x, current_y)  # 在当前位置点击
                            self.status_label.config(text="状态: 已触发", fg="green")
                        
                        # 状态重置条件
                        if self.consecutive_open >= self.cooldown_frames and self.eye_state:
                            self.eye_state = False
                            self.status_label.config(text="状态: 检测中", fg="blue")
                        
                        # 绘制眼部区域
                        self.draw_eye_region(processed_frame, landmarks, self.LEFT_EYE_INDICES)
                        self.draw_eye_region(processed_frame, landmarks, self.RIGHT_EYE_INDICES)
                        
                        # 显示调试信息
                        cv2.putText(processed_frame, f"EAR: {avg_ear:.3f}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                        cv2.line(processed_frame, (0, int(self.ear_threshold*100)),
                                (200, int(self.ear_threshold*100)), (0,0,255), 2)
                    if results.multi_face_landmarks:
                        self.detect_head_pose(results, processed_frame)
                except Exception as e:
                    print(f"处理异常: {str(e)}")
                
                # 显示处理后的画面
                processed_img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                processed_img = Image.fromarray(processed_img)
                processed_imgtk = ImageTk.PhotoImage(image=processed_img)
                self.processed_label.imgtk = processed_imgtk
                self.processed_label.configure(image=processed_imgtk)

    def calculate_ear(self, landmarks, eye_indices):
        """EAR求解"""
        p1 = landmarks[eye_indices[0]]
        p2 = landmarks[eye_indices[1]]
        p3 = landmarks[eye_indices[2]]
        p4 = landmarks[eye_indices[3]]
        p5 = landmarks[eye_indices[4]]
        p6 = landmarks[eye_indices[5]]
        
        # 计算垂直距离
        v1 = np.linalg.norm([p2.x - p6.x, p2.y - p6.y])
        v2 = np.linalg.norm([p3.x - p5.x, p3.y - p5.y])
        
        # 计算水平距离
        h = np.linalg.norm([p1.x - p4.x, p1.y - p4.y])
        
        return (v1 + v2) / (2.0 * h)

    def draw_eye_region(self, frame, landmarks, indices):
        """绘制眼部轮廓"""
        points = []
        for idx in indices:
            lm = landmarks[idx]
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            points.append((x, y))
        cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0,255,0), thickness=1)

    def update_threshold(self, value):
        """实时更新阈值"""
        self.ear_threshold = float(value)
        self.ear_label.config(text=f"EAR:阈值(0.20-0.30内调整，因人而异): {self.ear_threshold:.2f}")

    def toggle_detection(self):
        """切换检测状态"""
        self.running = not self.running
        if self.running:
            self.btn_switch.config(text="停止检测")
            self.status_label.config(text="状态: 检测中", fg="blue")
            threading.Thread(target=self.process_frame, daemon=True).start()
        else:
            self.btn_switch.config(text="启动检测")
            self.status_label.config(text="状态: 未激活", fg="gray")

    def on_closing(self):
        """关闭窗口时的清理操作"""
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()
    def detect_head_pose(self, results, frame):
        """头部姿态检测逻辑"""
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 获取2D关键点
        indexes = [1, 152, 33, 263, 61, 291]
        image_points = np.array([[
            landmarks[idx].x * self.img_width,
            landmarks[idx].y * self.img_height] for idx in indexes
        ], dtype=np.float64)
        
        # 计算头部姿态
        success, rotation_vector, _ = cv2.solvePnP(
            self.model_points, image_points,
            self.camera_matrix, self.dist_coeffs
        )
        
        if success:
            # 转换旋转向量为欧拉角
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            euler_angles = self.get_euler_angles(rotation_matrix)
            
            pitch = euler_angles[0]
            yaw = euler_angles[1]
            roll = euler_angles[2]
            
            # 更新显示信息
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            
            # 抬头检测逻辑
            if abs(roll) < self.roll_threshold:  # 排除转头动作
                if pitch > self.pitch_threshold:
                    self.nod_counter += 1
                    if self.nod_counter >= self.nod_required_frames:
                        self.lock_computer()
                else:
                    self.nod_counter = 0

    def get_euler_angles(self, R):
        """欧拉角（XYZ顺序）"""
        sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
            
        return np.degrees([x, y, z])

    def lock_computer(self):
        """锁定电脑，MAC自行修改"""
        current_time = time.time()
        if current_time - self.last_lock_time > self.lock_cooldown:
            ctypes.windll.user32.LockWorkStation()
            self.last_lock_time = current_time
            self.nod_counter = 0
            self.status_label.config(text="状态: 已锁定", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = BlinkControlApp(root)
    root.mainloop()