# -*- coding: utf-8 -*-
"""
颈部咽喉追踪系统 v2 - 模块化版本

本包提供了完整的颈部咽喉追踪功能，包括：
- 相机模块 (camera): Orbbec 深度相机支持和模拟相机
- 姿态检测 (openpose_utils): OpenPose 人体姿态检测
- 目标检测 (yolo_detector): YOLOv5 目标检测
- 数据追踪 (tracker_data): 3D 位置追踪和滤波
- 配置管理 (config): 系统配置加载和保存

使用示例:
    from neck_throat_v2 import main
    main.main()

或者单独导入需要的模块:
    from neck_throat_v2.camera import OrbbecCameraSource
    from neck_throat_v2.tracker_data import TrackerData
"""

__version__ = "2.0.0"
__author__ = "NC-Code Team"

# 相机模块导出
from .camera import (
    OrbbecCameraSource,
    SimulatedCameraSource,
    get_depth_at_point,
)

# OpenPose 姿态检测模块导出
from .openpose_utils import (
    get_neck_keypoints,
    get_face_keypoints,
    get_neck_region_bbox,
    init_openpose,
    process_frame,
)

# YOLO 目标检测模块导出
from .yolo_detector import (
    letterbox,
    detect_yolo,
    load_yolo_model,
    init_yolo,
)

# 数据追踪模块导出
from .tracker_data import (
    TrackerData,
)

# 配置管理模块导出
from .config import (
    DEFAULT_CONFIG,
    load_config,
    save_config,
    apply_config_to_tracker,
    print_config_summary,
)

# 定义公开的模块列表
__all__ = [
    # 相机模块
    "OrbbecCameraSource",
    "SimulatedCameraSource",
    "get_depth_at_point",
    
    # OpenPose 模块
    "get_neck_keypoints",
    "get_face_keypoints",
    "get_neck_region_bbox",
    "init_openpose",
    "process_frame",
    
    # YOLO 模块
    "letterbox",
    "detect_yolo",
    "load_yolo_model",
    "init_yolo",
    
    # 数据追踪模块
    "TrackerData",
    
    # 配置模块
    "DEFAULT_CONFIG",
    "load_config",
    "save_config",
    "apply_config_to_tracker",
    "print_config_summary",
]
