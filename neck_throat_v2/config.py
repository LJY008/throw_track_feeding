# -*- coding: utf-8 -*-
"""
配置管理模块 - 加载和保存配置
从 neck_throat_track.py 分离出的配置相关代码
"""

import os
import json


# 默认配置
DEFAULT_CONFIG = {
    # 检测间隔设置
    "openpose_interval": 3,
    "yolo_interval": 2,
    "plot_interval": 10,
    
    # 显示设置
    "display_scale": 0.7,
    
    # OpenPose参数
    "openpose_net_resolution": "256x256",
    "openpose_number_people_max": 1,
    
    # YOLO参数
    "yolo_imgsz": 416,
    "yolo_conf_threshold": 0.25,
    "yolo_iou_threshold": 0.45,
    
    # SORT追踪器参数
    "sort_max_age": 20,
    "sort_min_hits": 3,
    "sort_iou_threshold": 0.3,
    
    # TrackerData参数
    "tracker_window_size": 600,  # 增大到600秒（10分钟），避免数据被过早丢弃
    "tracker_id_timeout": 2.0,
    "tracker_max_points": 500,
    
    # 相机设置
    "camera_rgb_width": 640,
    "camera_rgb_height": 480,
    "camera_depth_width": 640,
    "camera_depth_height": 576,
    "camera_fps": 30,
    
    # 滤波参数
    "neck_buffer_size": 7,
    "neck_smoothing_alpha": 0.2,
    "depth_smoothing_alpha": 0.5,
    "neck_buffer": 7,
    "neck_alpha": 0.2,
    "depth_alpha": 0.5,
    "bilateral_filter_enabled": True,
    "bilateral_small_threshold_y": 2.0,
    "bilateral_large_threshold_y": 50.0,
    "bilateral_small_threshold_depth": 3.0,
    "bilateral_large_threshold_depth": 80.0,
    "neck_filter_enabled": True,
    
    # 身体运动检测参数
    "body_tilt_detection_enabled": True,  # 是否启用基于人体框架倾斜度的检测
    "line_slope_window": 60,
    "slope_stable_threshold": 0.15,
    "body_motion_grace_frames": 30,
    "baseline_adapt_alpha": 0.001,
    
    # QV波动率检测参数
    "qv_enabled": False,
    "qv_window_duration": 1.0,
    "qv_kernel_bandwidth": 0.5,
    "qv_volatility_threshold": 1.5,
    "qv_low_volatility_threshold": 0.3,
    "qv_suppress_slow_motion": True,
    
    # QV检测高级参数
    "qv_sampling_rate": 30.0,
    "qv_high_threshold": 1.5,
    "qv_h2_threshold": -2.0,
    "qv_suppression_window": 1.0,
    "qv_k0": 5,
    "qv_min_displacement": 1.0,
    "qv_verify_window": 1,
    "qv_invert_z": True,
    "qv_zero_threshold": 1e-10,
    "qv_w1": 0.7,
    "qv_w2": 0.1,
    "qv_w3": 0.2,
    "qv_min_std": 0.3,
    "qv_min_total_var": 0.5,
}


def load_config(config_file="tracker_config.json"):
    """加载配置文件
    
    参数:
        config_file: 配置文件路径
    
    返回:
        config: 配置字典
    """
    config = DEFAULT_CONFIG.copy()
    
    # 如果是绝对路径且文件存在，直接使用
    if os.path.isabs(config_file) and os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
                print(f"✅ 已加载配置文件: {config_file}")
            return config
        except Exception as e:
            print(f"⚠️ 配置文件加载失败: {e}，使用默认配置")
            return config
    
    # 尝试查找配置文件的多个可能位置
    # 1. 直接使用传入的路径（相对当前工作目录）
    # 2. 在 config.py 所在目录查找
    # 3. 在 config.py 所在目录的父目录查找（项目根目录）
    # 4. 相对当前工作目录查找
    possible_paths = [
        config_file,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_file),
        os.path.join(os.getcwd(), config_file)  # 添加当前工作目录
    ]

    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break

    if found_path:
        try:
            with open(found_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                # 合并配置，保留默认值作为fallback
                config.update(loaded_config)
                print(f"✅ 已加载配置文件: {found_path}")
        except Exception as e:
            print(f"⚠️ 配置文件加载失败: {e}，使用默认配置")
    else:
        # 显示调试信息，帮助排查问题
        print(f"⚠️ 未找到配置文件 {config_file}")
        print(f"   当前工作目录: {os.getcwd()}")
        print(f"   已尝试的路径: {possible_paths[0]}, {possible_paths[1]}, ...")
        print(f"   使用默认配置")
    
    return config


def save_config(config, config_file="tracker_config.json"):
    """保存配置到文件
    
    参数:
        config: 配置字典
        config_file: 配置文件路径
    
    返回:
        success: 是否成功
    """
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"✅ 配置已保存到: {config_file}")
        return True
    except Exception as e:
        print(f"❌ 配置保存失败: {e}")
        return False


def apply_config_to_tracker(tracker_data, config):
    """将配置应用到TrackerData对象
    
    参数:
        tracker_data: TrackerData对象
        config: 配置字典
    """
    # 滤波参数
    tracker_data.neck_buffer_size = config.get("neck_buffer_size", 7)
    tracker_data.neck_smoothing_alpha = config.get("neck_smoothing_alpha", 0.2)
    tracker_data.depth_smoothing_alpha = config.get("depth_smoothing_alpha", 0.5)
    tracker_data.bilateral_filter_enabled = config.get("bilateral_filter_enabled", True)
    tracker_data.bilateral_small_threshold_y = config.get("bilateral_small_threshold_y", 2.0)
    tracker_data.bilateral_large_threshold_y = config.get("bilateral_large_threshold_y", 50.0)
    tracker_data.bilateral_small_threshold_depth = config.get("bilateral_small_threshold_depth", 3.0)
    tracker_data.bilateral_large_threshold_depth = config.get("bilateral_large_threshold_depth", 80.0)
    tracker_data.neck_filter_enabled = config.get("neck_filter_enabled", True)
    
    # 身体运动检测参数
    tracker_data.body_tilt_detection_enabled = config.get("body_tilt_detection_enabled", True)
    tracker_data.line_slope_window = config.get("line_slope_window", 60)
    tracker_data.slope_stable_threshold = config.get("slope_stable_threshold", 0.15)
    tracker_data.body_motion_grace_frames = config.get("body_motion_grace_frames", 30)
    tracker_data.baseline_adapt_alpha = config.get("baseline_adapt_alpha", 0.001)
    
    # QV波动率检测参数
    tracker_data.qv_enabled = config.get("qv_enabled", False)
    tracker_data.qv_window_duration = config.get("qv_window_duration", 1.0)
    tracker_data.qv_kernel_bandwidth = config.get("qv_kernel_bandwidth", 0.5)
    tracker_data.qv_volatility_threshold = config.get("qv_volatility_threshold", 1.5)
    tracker_data.qv_low_volatility_threshold = config.get("qv_low_volatility_threshold", 0.3)
    tracker_data.qv_suppress_slow_motion = config.get("qv_suppress_slow_motion", True)
    
    # ID超时
    tracker_data.id_timeout = config.get("tracker_id_timeout", 2.0)
    
    print("✅ 配置已应用到TrackerData")


def print_config_summary(config):
    """打印配置摘要"""
    print("=" * 60)
    print("当前配置摘要:")
    print("=" * 60)
    print(f"  检测间隔: OpenPose={config.get('openpose_interval')}帧, YOLO={config.get('yolo_interval')}帧")
    print(f"  显示缩放: {config.get('display_scale')}")
    print(f"  OpenPose分辨率: {config.get('openpose_net_resolution')}")
    print(f"  YOLO尺寸: {config.get('yolo_imgsz')}, 置信度: {config.get('yolo_conf_threshold')}")
    print(f"  SORT: max_age={config.get('sort_max_age')}, min_hits={config.get('sort_min_hits')}")
    print(f"  滤波: neck_buffer={config.get('neck_buffer_size')}, neck_alpha={config.get('neck_smoothing_alpha')}")
    print(f"  双边滤波: {'启用' if config.get('bilateral_filter_enabled') else '禁用'}")
    print(f"  身体倾斜度检测: {'启用' if config.get('body_tilt_detection_enabled') else '禁用'}")
    print(f"  QV检测: {'启用' if config.get('qv_enabled') else '禁用'}")
    print("=" * 60)
