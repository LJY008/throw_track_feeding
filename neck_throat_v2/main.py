# -*- coding: utf-8 -*-
"""
颈部喉部追踪系统 - 主程序入口
结合 OpenPose 颈部检测和 Orbbec 深度相机
使用颈部关键点作为坐标原点，追踪相对位移（Y, Depth）

从 neck_throat_track.py 重构分离
"""

import os
import sys
import time
import argparse
import threading
from queue import Queue, Empty

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk  # 用于在Tkinter中显示OpenCV图像

# 解决 OpenMP 多副本冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 获取脚本目录
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)

# 添加父目录到路径（以便导入其他模块）
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入本地模块
from neck_throat_v2.camera import (
    OrbbecCameraSource, SimulatedCameraSource, 
    get_depth_at_point, ORBBEC_AVAILABLE
)
from neck_throat_v2.openpose_utils import (
    get_neck_keypoints, get_face_keypoints, get_neck_region_bbox,
    init_openpose, OPENPOSE_AVAILABLE, op
)
from neck_throat_v2.yolo_detector import (
    detect_yolo, load_yolo_model, YOLOV5_AVAILABLE
)
from neck_throat_v2.tracker_data import TrackerData
from neck_throat_v2.config import (
    load_config, apply_config_to_tracker, print_config_summary, DEFAULT_CONFIG
)

# 导入SORT追踪器
SORT_AVAILABLE = False
Sort = None
try:
    from sort import Sort
    SORT_AVAILABLE = True
    print("✅ SORT 追踪器已导入")
except ImportError as e:
    print(f'⚠️ SORT 追踪器不可用: {e}')


class QVDetectionThread:
    """QV检测的独立线程 - 在后台线程中执行完整的QV检测计算"""
    def __init__(self, qv_gui, tracker_data):
        self.qv_gui = qv_gui
        self.tracker_data = tracker_data
        self.running = False
        self.thread = None
        self.update_interval = 2.0  # 增加到2秒，减少检测频率
        self.last_update_time = 0
        self.detection_count = 0
        # 用于传递检测结果（包含完整的result对象）
        self.result_queue = Queue(maxsize=2)
        # 用于从主线程传递任务（包含参数和数据）
        self.task_queue = Queue(maxsize=2)
        # 标记是否正在执行检测，避免重复计算
        self._detecting = False
        # 缓存的检测器参数（由主线程更新）
        self._cached_params = {}
        
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True, name="QV-Detection")
        self.thread.start()
        print("✅ QV检测线程已启动（后台计算模式）")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print(f"✅ QV检测线程已停止（共运行{self.detection_count}次检测）")
    
    def submit_task(self, timestamps, y_disps, depth_disps, params):
        """从主线程提交检测任务（包含参数）"""
        if self._detecting:
            return False
        
        task = {
            'timestamps': timestamps,
            'y_disps': y_disps,
            'depth_disps': depth_disps,
            'params': params
        }
        
        try:
            # 清空旧任务
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except:
                    break
            self.task_queue.put(task, block=False)
            return True
        except:
            return False
        
    def _worker(self):
        """后台工作线程：执行完整的QV检测计算"""
        while self.running:
            try:
                # 从任务队列获取任务
                try:
                    task = self.task_queue.get(timeout=0.2)
                except Empty:
                    continue
                
                self._detecting = True
                current_time = time.time()
                
                try:
                    timestamps_arr = task['timestamps']
                    y_disps_arr = task['y_disps']
                    depth_disps_arr = task['depth_disps']
                    params = task['params']
                    
                    # 使用主线程传递的参数更新检测器
                    detector = self.qv_gui.detector
                    detector.fs = params['fs']
                    detector.dt = 1.0 / detector.fs
                    detector.h = params['h']
                    detector.H = params['H']
                    detector.H2 = params['H2']
                    detector.suppression_window = params['suppression_window']
                    detector.k0 = params['k0']
                    detector.min_displacement_change = params['min_displacement_change']
                    detector.window_size = params['window_size']
                    detector.w1 = params['w1']
                    detector.w2 = params['w2']
                    detector.w3 = params['w3']
                    
                    # 执行检测计算（这是耗时操作，在后台线程执行）
                    # 使用 silent=True 避免大量打印影响性能
                    result = detector.process_data(
                        timestamps_arr, 
                        depth_disps_arr, 
                        signal_data_y=y_disps_arr,
                        invert_z=params['invert_z'],
                        silent=True
                    )
                    
                    # 将结果放入队列
                    result_package = {
                        'timestamps': timestamps_arr,
                        'y_disps': y_disps_arr,
                        'depth_disps': depth_disps_arr,
                        'result': result,
                        'time': current_time
                    }
                    
                    # 非阻塞放入队列
                    if self.result_queue.full():
                        try:
                            self.result_queue.get_nowait()
                        except:
                            pass
                    self.result_queue.put(result_package, block=False)
                    self.detection_count += 1
                    
                except Exception as e:
                    print(f"⚠️ QV后台检测错误: {e}")
                finally:
                    self._detecting = False
                
                self.last_update_time = current_time
                
            except Exception as e:
                print(f"⚠️ QV检测线程错误: {e}")
                self._detecting = False
                time.sleep(0.5)
    
    def get_result(self):
        """获取检测结果（非阻塞）"""
        try:
            return self.result_queue.get_nowait()
        except:
            return None


def main():
    parser = argparse.ArgumentParser(description="颈部喉部追踪系统 v2")
    
    # 使用绝对路径（基于父目录）
    default_model_folder = os.path.join(parent_dir, "openpose_model")
    default_yolo_weights = os.path.join(parent_dir, "yolov5/runs/train/my_custom_train8/weights/best.pt")
    default_config = os.path.join(parent_dir, "tracker_config.json")
    
    parser.add_argument("--model_folder", default=default_model_folder, help="OpenPose模型文件夹路径")
    parser.add_argument("--yolo_weights", default=default_yolo_weights, 
                       help="YOLO权重文件路径")
    parser.add_argument("--net_resolution", default="256x256", help="OpenPose网络输入分辨率")
    parser.add_argument("--conf_thres", type=float, default=0.4, help="YOLO置信度阈值")
    parser.add_argument("--iou_thres", type=float, default=0.7, help="YOLO IoU阈值")
    parser.add_argument("--yolo_size", type=int, default=416, help="YOLO输入尺寸")
    parser.add_argument("--simulate", action="store_true", help="使用模拟模式")
    parser.add_argument("--config", default=default_config, help="配置文件路径")
    args = parser.parse_args()

    # 打印各模块可用性状态
    print("=" * 60)
    print("颈部喉部追踪系统 v2 启动")
    print("=" * 60)
    print("模块状态检测:")
    print(f"   Orbbec SDK: {'✅ 可用' if ORBBEC_AVAILABLE else '❌ 不可用'}")
    print(f"   OpenPose:   {'✅ 可用' if OPENPOSE_AVAILABLE else '❌ 不可用'}")
    print(f"   YOLOv5:     {'✅ 可用' if YOLOV5_AVAILABLE else '❌ 不可用'}")
    print(f"   SORT:       {'✅ 可用' if SORT_AVAILABLE else '❌ 不可用'}")

    # 检测是否需要模拟模式
    # 修改逻辑：只有当相机不可用时才强制模拟模式
    # OpenPose和YOLOv5不可用时可以降级运行（只显示画面，不进行检测）
    simulation_mode = args.simulate or not ORBBEC_AVAILABLE
    
    if simulation_mode:
        print("\n🎮 模拟模式启动")
        if args.simulate:
            print("   原因: 用户指定 --simulate 参数")
        if not ORBBEC_AVAILABLE:
            print("   原因: Orbbec SDK 不可用")
        
        camera = SimulatedCameraSource()
        camera.initialize()
        opWrapper = None
        model = None
        device = None
    else:
        # 正常模式：初始化真实设备
        print("\n正在初始化 Orbbec 相机...")
        camera = OrbbecCameraSource()
        if not camera.initialize():
            print("⚠️ 无法初始化 Orbbec 相机，切换到模拟模式")
            simulation_mode = True
            camera = SimulatedCameraSource()
            camera.initialize()
        else:
            print("✅ Orbbec 相机初始化成功")
        
        # 初始化 OpenPose（可选，不影响相机运行）
        opWrapper = None
        if not simulation_mode and OPENPOSE_AVAILABLE:
            print("正在初始化 OpenPose...")
            print(f"   模型路径: {args.model_folder}")
            print(f"   路径存在: {os.path.exists(args.model_folder)}")
            opWrapper = init_openpose(args.model_folder, args.net_resolution)
            if opWrapper is None:
                print("⚠️ OpenPose 初始化失败，将跳过姿态检测")
            else:
                print("✅ OpenPose 初始化成功")
        elif not OPENPOSE_AVAILABLE:
            print("⚠️ OpenPose 不可用，将跳过姿态检测")
        
        # 初始化 YOLOv5（可选，不影响相机运行）
        model = None
        device = None
        if not simulation_mode and YOLOV5_AVAILABLE:
            print("正在加载 YOLOv5 模型...")
            print(f"   权重路径: {args.yolo_weights}")
            print(f"   路径存在: {os.path.exists(args.yolo_weights)}")
            model, device = load_yolo_model(args.yolo_weights)
            if model is None:
                print("⚠️ YOLOv5 模型加载失败，将跳过目标检测")
        elif not YOLOV5_AVAILABLE:
            print("⚠️ YOLOv5 不可用，将跳过目标检测")

    # 加载配置
    config = load_config(args.config)
    print_config_summary(config)

    # 创建 SORT 追踪器
    if SORT_AVAILABLE and Sort is not None:
        sort_tracker = Sort(
            max_age=config.get("sort_max_age", 20),
            min_hits=config.get("sort_min_hits", 3),
            iou_threshold=config.get("sort_iou_threshold", 0.3)
        )
    else:
        sort_tracker = None
        print("⚠️ SORT追踪器不可用")
    
    # 创建数据追踪器
    tracker_data = TrackerData(window_size=config.get("tracker_window_size", 300))
    apply_config_to_tracker(tracker_data, config)
    
    # 设置相机内参
    if hasattr(camera, 'depth_intrinsics') and camera.depth_intrinsics is not None:
        tracker_data.depth_intrinsics = camera.depth_intrinsics
        tracker_data.color_intrinsics = camera.color_intrinsics
        tracker_data.extrinsic = camera.extrinsic
        print("✅ 已设置相机内参到tracker_data")
    
    # 创建 Tkinter 窗口
    root = tk.Tk()
    root.title("颈喉追踪 + QV实时监测 v2")
    root.geometry("1800x950")
    
    quit_flag = {'value': False}
    
    # ========== 创建相机预览窗口（独立Toplevel窗口） ==========
    camera_window = tk.Toplevel(root)
    camera_window.title("相机预览 - Neck Throat Tracking v2")
    camera_window.geometry("720x560")
    camera_window.resizable(True, True)
    
    # 相机画面标签
    camera_label = ttk.Label(camera_window)
    camera_label.pack(fill=tk.BOTH, expand=True)
    
    # 状态栏
    camera_status_frame = ttk.Frame(camera_window)
    camera_status_frame.pack(fill=tk.X, side=tk.BOTTOM)
    camera_status_var = tk.StringVar(value="等待数据...")
    ttk.Label(camera_status_frame, textvariable=camera_status_var).pack(side=tk.LEFT, padx=5)
    
    # 快捷键提示
    ttk.Label(camera_status_frame, text="快捷键: [空格]记录金标准 | [R]重置原点 | [S]保存图像 | [Q]退出", 
             foreground="gray").pack(side=tk.RIGHT, padx=5)
    
    # 保存相机画面引用（用于更新）
    camera_photo = {'image': None}  # 防止被垃圾回收
    
    # 相机窗口关闭事件处理（稍后与主窗口一起统一设置）
    # camera_window.protocol("WM_DELETE_WINDOW", ...) 在后面设置
    
    # 绑定相机窗口的键盘事件
    def on_camera_key(event):
        if event.char == 'q' or event.keysym == 'Escape':
            quit_flag['value'] = True
        elif event.char == 's':
            # 保存图像功能在process_frame中处理
            camera_window.event_generate('<<SaveImage>>')
        elif event.char == 'r':
            camera_window.event_generate('<<ResetOrigin>>')
        elif event.char == ' ':
            camera_window.event_generate('<<RecordGold>>')
    
    camera_window.bind('<KeyPress>', on_camera_key)
    camera_window.focus_set()  # 默认聚焦到相机窗口
    
    # ========== 导入QV监测GUI ==========
    try:
        from neck_throat_v2.qv_realtime_monitor import QVMonitorGUI
        qv_gui = QVMonitorGUI(root)
        qv_gui.file_label.config(text="实时数据模式", foreground="green")
        qv_gui.realtime_var.set(True)
        qv_gui.realtime_update = True
        
        QV_GUI_AVAILABLE = True
        print("✅ QV监测GUI已加载")
    except ImportError as e:
        print(f"⚠️ 无法导入QV监测GUI: {e}")
        QV_GUI_AVAILABLE = False
        qv_gui = None
    
    # 辅助函数：检查是否暂停（兼容 qv_gui 不可用的情况）
    def is_paused():
        if qv_gui is not None:
            return qv_gui.is_paused()
        return False
    
    # 暂停时间累计（用于保证恢复后时间戳连续）
    pause_state = {
        'total_paused_duration': 0.0,  # 累计暂停时长
        'pause_start_time': None,      # 暂停开始时间
        'was_paused': False            # 上一帧是否暂停
    }
    
    # 暂停状态改变回调（用于记录暂停/恢复时间）
    def on_pause_changed(is_now_paused):
        if is_now_paused:
            # 开始暂停，记录暂停开始时间
            pause_state['pause_start_time'] = time.time()
        else:
            # 恢复追踪，累加暂停时长
            if pause_state['pause_start_time'] is not None:
                paused_duration = time.time() - pause_state['pause_start_time']
                pause_state['total_paused_duration'] += paused_duration
                print(f"   暂停了 {paused_duration:.1f} 秒，累计暂停 {pause_state['total_paused_duration']:.1f} 秒")
                pause_state['pause_start_time'] = None
    
    # 设置暂停回调
    if qv_gui is not None:
        qv_gui.set_pause_callback(on_pause_changed)
    
    # 导出所有数据函数（追踪数据 + 检测结果）
    def export_all_data():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 1. 导出追踪数据
        # tracking_filename = f"tracking_data_{timestamp}.csv"
        # tracker_data.export_to_csv(tracking_filename)
        # print(f"✅ 追踪数据已导出: {tracking_filename}")
        
        # 2. 导出检测结果（如果QV检测已运行）
        if qv_gui is not None:
            try:
                qv_gui.export_detection_results_silent(timestamp)
            except Exception as e:
                print(f"⚠️ 检测结果导出失败: {e}")
    
    # 设置导出回调
    if qv_gui is not None:
        qv_gui.set_export_callback(export_all_data)
    
    # 重置所有数据函数
    def reset_all_data():
        nonlocal sort_tracker
        
        # 1. 重置追踪数据
        tracker_data.reset()
        print("✅ tracker_data 已重置")
        
        # 2. 重置SORT追踪器
        if SORT_AVAILABLE and Sort is not None:
            sort_tracker = Sort(
                max_age=config.get("sort_max_age", 20),
                min_hits=config.get("sort_min_hits", 3),
                iou_threshold=config.get("sort_iou_threshold", 0.3)
            )
            print("✅ SORT追踪器已重置")
        
        # 3. 重置暂停时间累计
        pause_state['total_paused_duration'] = 0.0
        pause_state['pause_start_time'] = None
        pause_state['was_paused'] = False
        
        # 4. 清空金标准记录
        gold_standard_timestamps.clear()
        gold_standard_data.clear()
        print("✅ 金标准记录已清空")
        
        print("=" * 40)
        print("🔄 所有数据已重置，可重新开始采集")
        print("=" * 40)
    
    # 设置重置回调
    if qv_gui is not None:
        qv_gui.set_reset_callback(reset_all_data)
    
    # 绑定键盘事件（仅保留退出功能）
    def on_key_press(event):
        if event.char == 'q' or event.keysym == 'Escape':
            quit_flag['value'] = True
            print("\n收到退出命令")
    
    root.bind('<KeyPress>', on_key_press)

    # 性能参数
    openpose_interval = config.get("openpose_interval", 3)
    yolo_interval = config.get("yolo_interval", 2)
    display_scale = config.get("display_scale", 0.7)
    
    # 多线程队列
    openpose_queue = Queue(maxsize=2)
    yolo_queue = Queue(maxsize=2)
    frame_for_openpose = Queue(maxsize=2)
    frame_for_yolo = Queue(maxsize=2)
    
    # 缓存检测结果
    last_neck_info = None
    last_face_info = None
    last_neck_region_bbox = None
    last_detections = []
    
    # 工作线程运行标志
    worker_running = {'openpose': True, 'yolo': True}
    
    # 工作线程
    openpose_thread = None
    yolo_thread = None
    
    # OpenPose 工作线程（仅当 opWrapper 可用时启动）
    if not simulation_mode and opWrapper is not None:
        def openpose_worker():
            while worker_running['openpose']:
                try:
                    frame_data = frame_for_openpose.get(timeout=0.1)
                    if frame_data is None:
                        break
                    
                    color_frame, frame_num = frame_data
                    datum = op.Datum()
                    datum.cvInputData = color_frame
                    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                    
                    neck_info = get_neck_keypoints(datum.poseKeypoints)
                    face_info = get_face_keypoints(datum.faceKeypoints)
                    neck_region_bbox = get_neck_region_bbox(neck_info, face_info)
                    output_img = datum.cvOutputData.copy() if datum.cvOutputData is not None else None
                    
                    openpose_queue.put((neck_info, face_info, neck_region_bbox, output_img, frame_num))
                except Empty:
                    continue
                except Exception as e:
                    if worker_running['openpose']:
                        print(f"OpenPose 线程错误: {e}")
        
        openpose_thread = threading.Thread(target=openpose_worker, daemon=True, name="OpenPose")
        openpose_thread.start()
        print("✅ OpenPose 工作线程已启动")
    
    # YOLO 工作线程（仅当 model 可用时启动）
    if not simulation_mode and model is not None:
        def yolo_worker():
            while worker_running['yolo']:
                try:
                    frame_data = frame_for_yolo.get(timeout=0.1)
                    if frame_data is None:
                        break
                    
                    color_frame, frame_num = frame_data
                    detections = detect_yolo(color_frame, model, device, 
                                           imgsz=args.yolo_size,
                                           conf_thres=args.conf_thres, 
                                           iou_thres=args.iou_thres)
                    yolo_queue.put((detections, frame_num))
                except Empty:
                    continue
                except Exception as e:
                    if worker_running['yolo']:
                        print(f"YOLO 线程错误: {e}")
        
        yolo_thread = threading.Thread(target=yolo_worker, daemon=True, name="YOLO")
        yolo_thread.start()
        print("✅ YOLO 工作线程已启动")
    
    # 启动QV检测线程
    qv_detection_thread = None
    if qv_gui is not None:
        qv_detection_thread = QVDetectionThread(qv_gui, tracker_data)
        qv_detection_thread.start()
    
    # 金标准记录
    gold_standard_timestamps = []
    gold_standard_data = []
    if qv_gui is not None:
        qv_gui.gold_standard_timestamps = gold_standard_timestamps
        qv_gui.gold_standard_data = gold_standard_data

    print("=" * 60)
    print("🎯 开始追踪...")
    if not simulation_mode:
        mode_info = "相机模式"
        if opWrapper is not None and model is not None:
            mode_info += " (OpenPose + YOLO)"
        elif opWrapper is not None:
            mode_info += " (仅OpenPose)"
        elif model is not None:
            mode_info += " (仅YOLO)"
        else:
            mode_info += " (仅显示画面)"
        print(f"   运行模式: {mode_info}")
    else:
        print("   运行模式: 模拟模式")
    print("   按 'q' 键退出, 's' 键保存图像, 'r' 键重置原点")
    print("   空格键记录金标准吞咽事件")
    print("   使用界面底部按钮进行暂停和导出操作")
    print("=" * 60)

    start_time = time.time()
    
    # 主循环状态（用于 Tkinter after 调度）
    main_loop_state = {
        'running': True,
        'frame_count': 0,
        'fps_time': time.time(),
        'fps': 0,
        'current_frame': None,  # 用于保存当前帧
        'current_time': 0.0,    # 当前有效时间
        'after_ids': []         # 保存所有 after 调度的ID
    }
    
    # ========== 自定义事件处理器 ==========
    def on_save_image(event):
        """保存图像"""
        if main_loop_state['current_frame'] is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, main_loop_state['current_frame'])
            print(f"保存图像: {filename}")
    
    def on_reset_origin(event):
        """重置原点"""
        nonlocal sort_tracker
        tracker_data.reset()
        if sort_tracker is not None and SORT_AVAILABLE:
            sort_tracker = Sort(
                max_age=config.get("sort_max_age", 20),
                min_hits=config.get("sort_min_hits", 3),
                iou_threshold=config.get("sort_iou_threshold", 0.3)
            )
        print("已重置原点和追踪器")
    
    def on_record_gold(event):
        """记录金标准"""
        gold_ts = main_loop_state['current_time']
        try:
            feeding_volume = qv_gui.current_feeding_volume.get() if qv_gui else 5.0
        except:
            feeding_volume = 5.0
        
        gold_standard_timestamps.append(gold_ts)
        gold_standard_data.append((gold_ts, feeding_volume))
        if qv_gui:
            qv_gui.gold_standard_timestamps = gold_standard_timestamps
            qv_gui.gold_standard_data = gold_standard_data
        print(f"🏅 金标准 #{len(gold_standard_timestamps)} @ {gold_ts:.3f}s")
    
    # 绑定自定义事件
    camera_window.bind('<<SaveImage>>', on_save_image)
    camera_window.bind('<<ResetOrigin>>', on_reset_origin)
    camera_window.bind('<<RecordGold>>', on_record_gold)

    def process_frame():
        """处理单帧的函数，由 Tkinter after 调度"""
        nonlocal sort_tracker, gold_standard_timestamps, gold_standard_data
        nonlocal last_neck_info, last_face_info, last_neck_region_bbox, last_detections
        
        # 检查窗口是否还存在
        try:
            if not main_loop_state['running'] or quit_flag['value']:
                return
            
            if not root.winfo_exists():
                return
        except:
            return
        
        try:
            color_image, depth_image = camera.get_frames()
            if color_image is None:
                # 没有帧，稍后重试
                after_id = root.after(10, process_frame)
                main_loop_state['after_ids'].append(after_id)
                return

            # 计算有效时间（扣除暂停时长）
            current_time = time.time() - start_time - pause_state['total_paused_duration']
            
            # 如果当前正在暂停，还需要扣除当前暂停的时间
            if is_paused() and pause_state['pause_start_time'] is not None:
                current_time -= (time.time() - pause_state['pause_start_time'])
            
            # 保存到状态（用于事件处理器）
            main_loop_state['current_time'] = current_time
            
            # 模拟模式数据处理
            if simulation_mode:
                if isinstance(camera, SimulatedCameraSource) and not is_paused():
                    sim_time, y_disp, z_disp = camera.get_simulated_target_position()
                    if sim_time is not None:
                        if tracker_data.neck_origin is None:
                            tracker_data.set_neck_origin(320, 240, 1000)
                        
                        sim_x = 320
                        sim_y = 240 + y_disp * 5
                        sim_depth = 1000 + z_disp * 10
                        tracker_data.add_position(1, sim_x, sim_y, sim_depth, current_time)
                
                output_image = color_image.copy()
            else:
                # 提交帧到工作线程（暂停时不提交新帧）
                if not is_paused():
                    # 只有当 OpenPose 线程存在时才提交帧
                    if openpose_thread is not None and main_loop_state['frame_count'] % openpose_interval == 0:
                        if not frame_for_openpose.full():
                            frame_for_openpose.put((color_image.copy(), main_loop_state['frame_count']))
                    
                    # 只有当 YOLO 线程存在时才提交帧
                    if yolo_thread is not None and main_loop_state['frame_count'] % yolo_interval == 0:
                        if not frame_for_yolo.full():
                            frame_for_yolo.put((color_image.copy(), main_loop_state['frame_count']))
                
                # 获取检测结果（只有当对应线程存在时才尝试获取）
                if openpose_thread is not None:
                    try:
                        result = openpose_queue.get_nowait()
                        last_neck_info, last_face_info, last_neck_region_bbox, _, _ = result
                    except Empty:
                        pass
                
                if yolo_thread is not None:
                    try:
                        result = yolo_queue.get_nowait()
                        last_detections, _ = result
                    except Empty:
                        pass
                
                output_image = color_image.copy()
                neck_info = last_neck_info
                face_info = last_face_info
                neck_region_bbox = last_neck_region_bbox
                
                # 绘制颈部区域
                if neck_region_bbox is not None:
                    x1, y1, x2, y2 = neck_region_bbox
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # 获取颈部深度并设置原点
                if neck_info and 'neck' in neck_info:
                    neck_x, neck_y = neck_info['neck'][0], neck_info['neck'][1]
                    neck_depth = get_depth_at_point(depth_image, neck_x, neck_y)
                    
                    if neck_depth is not None:
                        tracker_data.set_neck_origin(neck_x, neck_y, neck_depth)
                    
                    cv2.circle(output_image, (neck_x, neck_y), 8, (0, 0, 255), -1)
                    cv2.putText(output_image, "Neck", (neck_x + 10, neck_y - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    if neck_depth is not None:
                        cv2.putText(output_image, f"{neck_depth:.0f}mm", (neck_x + 10, neck_y + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # SORT追踪
                tracked_objects = np.empty((0, 5))
                if sort_tracker is not None:
                    if len(last_detections) > 0:
                        dets = np.array(last_detections)
                        tracked_objects = sort_tracker.update(dets[:, :4])
                    else:
                        tracked_objects = sort_tracker.update(np.empty((0, 5)))
            
            # 处理追踪结果（暂停时不添加新位置数据，但仍绘制框）
            if not simulation_mode and len(tracked_objects) > 0 and tracker_data.neck_origin is not None:
                for trk in tracked_objects:
                    bbox = trk[:4]
                    track_id = int(trk[-1])
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # 只有未暂停时才记录数据
                    if not is_paused():
                        target_depth = get_depth_at_point(depth_image, center_x, center_y)
                        tracker_data.add_position(track_id, center_x, center_y, target_depth, current_time)
                    
                    if track_id == tracker_data.target_track_id:
                        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.circle(output_image, (center_x, center_y), 5, (0, 0, 255), -1)
                        cv2.putText(output_image, f"ID:{track_id}", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 计算 FPS
            main_loop_state['frame_count'] += 1
            if main_loop_state['frame_count'] % 10 == 0:
                main_loop_state['fps'] = 10 / (time.time() - main_loop_state['fps_time'])
                main_loop_state['fps_time'] = time.time()

            # 显示状态信息
            cv2.putText(output_image, f"FPS: {main_loop_state['fps']:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            gold_count = len(gold_standard_timestamps)
            cv2.putText(output_image, f"Gold: {gold_count} [SPACE]", (output_image.shape[1] - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
            
            # 显示暂停状态
            if is_paused():
                cv2.putText(output_image, "PAUSED", (output_image.shape[1] // 2 - 60, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(output_image, "Data recording stopped", 
                           (output_image.shape[1] // 2 - 120, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if tracker_data.neck_origin is not None:
                cv2.putText(output_image, "Origin: SET", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(output_image, "Origin: WAITING", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # ========== 在 Tkinter 中显示图像（替代 cv2.imshow） ==========
            display_image = cv2.resize(output_image, None, fx=display_scale, fy=display_scale, 
                                      interpolation=cv2.INTER_LINEAR)
            # OpenCV BGR -> RGB -> PIL Image -> ImageTk
            display_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(display_rgb)
            camera_photo['image'] = ImageTk.PhotoImage(image=pil_image)
            camera_label.config(image=camera_photo['image'])
            
            # 更新状态栏
            status_text = f"FPS: {main_loop_state['fps']:.1f} | 帧: {main_loop_state['frame_count']}"
            if is_paused():
                status_text += " | ⏸ 已暂停"
            camera_status_var.set(status_text)
            
            # ========== 保存当前帧用于保存功能 ==========
            main_loop_state['current_frame'] = output_image

            # 按键处理已通过 Tkinter 事件绑定实现（on_camera_key）
            # 不再使用 cv2.waitKey()

            # 更新QV GUI - 主线程提交任务，后台线程计算，主线程更新GUI
            if qv_detection_thread is not None and qv_gui is not None:
                # 1. 检查是否需要提交新任务（在主线程中读取参数和数据）
                current_loop_time = time.time()
                if (current_loop_time - qv_detection_thread.last_update_time >= qv_detection_thread.update_interval
                    and not qv_detection_thread._detecting
                    and qv_gui.realtime_update):
                    
                    # 获取展示用数据（滑动窗口，只保留最近一段时间）
                    timestamps_list = tracker_data.get_timestamps()
                    y_disps_list, depth_disps_list = tracker_data.get_displacements()
                    
                    # 同时获取全量数据（用于最终导出和分析）
                    full_timestamps_list = tracker_data.get_full_timestamps()
                    full_y_disps_list, full_depth_disps_list = tracker_data.get_full_displacements()
                    
                    if len(timestamps_list) >= 100:
                        # ========== UI展示窗口限制 ==========
                        # 只取最近 display_window 秒的数据用于UI绘图，减少matplotlib绑图压力
                        display_window_sec = getattr(tracker_data, 'display_window', 60)  # 默认60秒
                        
                        timestamps_arr = np.array(timestamps_list)
                        y_disps_arr = np.array(y_disps_list)
                        depth_disps_arr = np.array(depth_disps_list)
                        
                        # 计算时间窗口内的数据索引
                        if len(timestamps_arr) > 0:
                            current_max_time = timestamps_arr[-1]
                            window_start_time = current_max_time - display_window_sec
                            window_mask = timestamps_arr >= window_start_time
                            
                            # 只保留窗口内的数据用于UI展示
                            display_timestamps = timestamps_arr[window_mask]
                            display_y_disps = y_disps_arr[window_mask]
                            display_depth_disps = depth_disps_arr[window_mask]
                        else:
                            display_timestamps = timestamps_arr
                            display_y_disps = y_disps_arr
                            display_depth_disps = depth_disps_arr
                        
                        # 在主线程中读取Tkinter变量
                        params = {
                            'fs': qv_gui.fs_var.get(),
                            'h': qv_gui.h_var.get(),
                            'H': qv_gui.H_var.get(),
                            'H2': qv_gui.H2_var.get(),
                            'suppression_window': qv_gui.suppress_var.get(),
                            'k0': qv_gui.k0_var.get(),
                            'min_displacement_change': qv_gui.min_disp_var.get(),
                            'window_size': qv_gui.window_var.get(),
                            'w1': qv_gui.w1_var.get(),
                            'w2': qv_gui.w2_var.get(),
                            'w3': qv_gui.w3_var.get(),
                            'invert_z': qv_gui.invert_z_var.get()
                        }
                        
                        # 提交任务到后台线程（使用窗口内的数据进行检测和绑图）
                        if len(display_timestamps) >= 100:
                            qv_detection_thread.submit_task(
                                display_timestamps,
                                display_y_disps,
                                display_depth_disps,
                                params
                            )
                        qv_detection_thread.last_update_time = current_loop_time
                        
                        # 同时更新全量数据到 qv_gui（用于最终导出）
                        if len(full_timestamps_list) >= 100:
                            qv_gui.full_timestamps = np.array(full_timestamps_list)
                            qv_gui.full_signal_data_y = np.array(full_y_disps_list)
                            qv_gui.full_signal_data = np.array(full_depth_disps_list)
                
                # 2. 获取后台计算的结果并更新GUI
                result_package = qv_detection_thread.get_result()
                if result_package is not None:
                    try:
                        # 更新展示用数据
                        qv_gui.timestamps = result_package['timestamps']
                        qv_gui.signal_data_y = result_package['y_disps']
                        qv_gui.signal_data = result_package['depth_disps']
                        
                        # 直接使用后台计算好的检测结果
                        qv_gui.result = result_package['result']
                        
                        timestamps_arr = result_package['timestamps']
                        data_info = f"实时: {len(timestamps_arr)}点, {timestamps_arr[-1]-timestamps_arr[0]:.1f}s"
                        qv_gui.file_label.config(text=data_info, foreground="green")
                        
                        # 更新统计和图表
                        if qv_gui.result is not None:
                            qv_gui.update_statistics()
                            qv_gui.update_plots()
                    except Exception as e:
                        pass
            
            # 调度下一帧处理（约30fps）
            after_id = root.after(15, process_frame)
            main_loop_state['after_ids'].append(after_id)
        
        except Exception as e:
            print(f"帧处理错误: {e}")
            import traceback
            traceback.print_exc()
            # 出错后继续尝试
            after_id = root.after(100, process_frame)
            main_loop_state['after_ids'].append(after_id)
    
    def cleanup_and_exit():
        """清理资源并退出"""
        # 防止重复调用
        if hasattr(cleanup_and_exit, '_called'):
            return
        cleanup_and_exit._called = True
        
        print("\n正在停止...")
        
        # 0. 停止主循环并取消所有 after 调度
        main_loop_state['running'] = False
        
        # 取消所有待执行的 after 任务
        for after_id in main_loop_state['after_ids']:
            try:
                root.after_cancel(after_id)
            except:
                pass
        main_loop_state['after_ids'].clear()
        
        # 1. 首先设置运行标志为 False，让工作线程自行退出
        worker_running['openpose'] = False
        worker_running['yolo'] = False
        
        # 2. 停止 QV 检测线程
        if qv_detection_thread is not None:
            qv_detection_thread.stop()
        
        # 3. 发送终止信号并等待工作线程结束
        if openpose_thread is not None:
            try:
                frame_for_openpose.put(None, timeout=0.5)
            except:
                pass
            openpose_thread.join(timeout=1.0)
        if yolo_thread is not None:
            try:
                frame_for_yolo.put(None, timeout=0.5)
            except:
                pass
            yolo_thread.join(timeout=1.0)
        
        # 4. 释放相机资源
        try:
            camera.release()
        except:
            pass
        
        # 5. 关闭相机预览窗口
        try:
            if camera_window.winfo_exists():
                camera_window.destroy()
        except:
            pass
        
        # 5.5 关闭IMU窗口（如果存在）
        try:
            if imu_window is not None and imu_window.winfo_exists():
                imu_window.destroy()
        except:
            pass
        
        # 6. 退出 Tkinter
        try:
            if root.winfo_exists():
                root.quit()
                root.destroy()
        except:
            pass
        
        print("程序结束")
    
    # 窗口关闭事件处理
    def on_window_close():
        print("\n收到窗口关闭命令")
        main_loop_state['running'] = False
        quit_flag['value'] = True
        # 直接调用清理（避免 after 调度问题）
        cleanup_and_exit()
    
    root.protocol("WM_DELETE_WINDOW", on_window_close)
    
    # 同样处理相机窗口关闭
    def on_camera_window_close_updated():
        print("\n相机窗口关闭")
        on_window_close()
    
    camera_window.protocol("WM_DELETE_WINDOW", on_camera_window_close_updated)
    
    # ========== 创建IMU数据展示窗口 ==========
    imu_window = None
    imu_axes = None
    imu_canvas = None
    imu_lines = {}
    
    def create_imu_window():
        """创建IMU数据曲线展示窗口"""
        nonlocal imu_window, imu_axes, imu_canvas, imu_lines
        
        if qv_gui is None:
            return
        
        imu_window = tk.Toplevel(root)
        imu_window.title("IMU数据实时曲线")
        imu_window.geometry("800x600")
        
        # 创建matplotlib图表
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        fig.tight_layout(pad=3.0)
        
        imu_axes = axes
        
        # 上图：欧拉角
        axes[0].set_title('IMU 欧拉角')
        axes[0].set_xlabel('时间 (秒)')
        axes[0].set_ylabel('角度 (度)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(['Roll', 'Pitch', 'Yaw'])
        
        # 下图：加速度
        axes[1].set_title('IMU 加速度')
        axes[1].set_xlabel('时间 (秒)')
        axes[1].set_ylabel('加速度 (G)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(['Acc X', 'Acc Y', 'Acc Z'])
        
        # 初始化空线条
        imu_lines['roll'], = axes[0].plot([], [], 'r-', label='Roll', linewidth=1.5)
        imu_lines['pitch'], = axes[0].plot([], [], 'g-', label='Pitch', linewidth=1.5)
        imu_lines['yaw'], = axes[0].plot([], [], 'b-', label='Yaw', linewidth=1.5)
        
        imu_lines['acc_x'], = axes[1].plot([], [], 'r-', label='Acc X', linewidth=1.5)
        imu_lines['acc_y'], = axes[1].plot([], [], 'g-', label='Acc Y', linewidth=1.5)
        imu_lines['acc_z'], = axes[1].plot([], [], 'b-', label='Acc Z', linewidth=1.5)
        
        axes[0].legend()
        axes[1].legend()
        
        # 嵌入Tkinter
        imu_canvas = FigureCanvasTkAgg(fig, master=imu_window)
        imu_canvas.draw()
        imu_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 状态栏
        status_frame = ttk.Frame(imu_window)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        imu_status_var = tk.StringVar(value="等待IMU数据...")
        ttk.Label(status_frame, textvariable=imu_status_var).pack(side=tk.LEFT, padx=5)
        imu_window.imu_status_var = imu_status_var  # 保存引用
        
        # 关闭按钮
        ttk.Button(status_frame, text="关闭", command=imu_window.destroy).pack(side=tk.RIGHT, padx=5)
        
        # 窗口关闭时清理
        def on_imu_window_close():
            try:
                imu_window.destroy()
            except:
                pass
        
        imu_window.protocol("WM_DELETE_WINDOW", on_imu_window_close)
        
        print("✅ IMU数据展示窗口已创建")
    
    def update_imu_plot():
        """更新IMU数据曲线"""
        # 检查窗口是否存在
        try:
            if imu_window is None or not imu_window.winfo_exists():
                return
            
            if not main_loop_state['running']:
                return
        except:
            return
        
        if qv_gui is None or qv_gui.imu_reader is None:
            after_id = root.after(500, update_imu_plot)
            main_loop_state['after_ids'].append(after_id)
            return
        
        try:
            # 获取数据
            times, rolls, pitches, yaws = qv_gui.imu_reader.get_data()
            acc_times, acc_xs, acc_ys, acc_zs = qv_gui.imu_reader.get_acc_data()
            
            if len(times) > 0:
                # 更新欧拉角曲线
                imu_lines['roll'].set_data(times, rolls)
                imu_lines['pitch'].set_data(times, pitches)
                imu_lines['yaw'].set_data(times, yaws)
                
                # 自动调整X轴范围
                imu_axes[0].set_xlim(max(0, times[-1] - 30), times[-1] + 1)
                imu_axes[0].relim()
                imu_axes[0].autoscale_view(scalex=False, scaley=True)
                
                # 更新加速度曲线
                if len(acc_times) > 0:
                    imu_lines['acc_x'].set_data(acc_times, acc_xs)
                    imu_lines['acc_y'].set_data(acc_times, acc_ys)
                    imu_lines['acc_z'].set_data(acc_times, acc_zs)
                    
                    imu_axes[1].set_xlim(max(0, acc_times[-1] - 30), acc_times[-1] + 1)
                    imu_axes[1].relim()
                    imu_axes[1].autoscale_view(scalex=False, scaley=True)
                
                # 更新画布
                imu_canvas.draw_idle()
                
                # 更新状态
                status = f"数据点: {len(times)} | Roll: {rolls[-1]:.1f}° | Pitch: {pitches[-1]:.1f}° | Yaw: {yaws[-1]:.1f}°"
                imu_window.imu_status_var.set(status)
            else:
                imu_window.imu_status_var.set("等待IMU数据...")
        
        except Exception as e:
            print(f"更新IMU图表错误: {e}")
        
        # 继续更新
        after_id = root.after(200, update_imu_plot)
        main_loop_state['after_ids'].append(after_id)
    
    # 如果QV GUI可用，添加打开IMU窗口的按钮（可选）
    if qv_gui is not None:
        def toggle_imu_window():
            """切换IMU窗口显示"""
            if imu_window is None or not imu_window.winfo_exists():
                create_imu_window()
                after_id = root.after(100, update_imu_plot)
                main_loop_state['after_ids'].append(after_id)
            else:
                imu_window.destroy()
        
        # 保存引用供外部调用
        root.toggle_imu_window = toggle_imu_window
    
    # 启动帧处理循环
    after_id = root.after(100, process_frame)
    main_loop_state['after_ids'].append(after_id)
    
    # 启动 Tkinter 主事件循环（这会阻塞直到 root.quit() 被调用）
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n用户中断程序")
        main_loop_state['running'] = False
        cleanup_and_exit()


if __name__ == "__main__":
    main()
