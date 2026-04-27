# -*- coding: utf-8 -*-
"""
YOLO检测模块 - 目标检测相关函数
从 neck_throat_track.py 分离出的YOLO相关代码
"""

import os
import sys
import numpy as np
import cv2

# 获取脚本目录（父目录）
script_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# 添加 YOLOv5 路径
yolov5_path = os.path.join(script_dir, 'yolov5')
if yolov5_path not in sys.path:
    sys.path.insert(0, yolov5_path)

# 导入torch
TORCH_AVAILABLE = False
torch = None
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ torch 不可用")

# 导入YOLO相关
YOLOV5_AVAILABLE = False
non_max_suppression = None
scale_boxes = None
attempt_load = None
select_device = None
set_logging = None

try:
    from models.experimental import attempt_load
    from utils.general import check_img_size, non_max_suppression, set_logging, scale_boxes
    from utils.torch_utils import select_device
    YOLOV5_AVAILABLE = True
    print("✅ YOLOv5 已导入")
except ImportError as e:
    YOLOV5_AVAILABLE = False
    print(f'⚠️ YOLOv5 不可用: {e}')


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, 
              scaleFill=False, scaleup=True, stride=32):
    """调整图像大小以适应模型输入"""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def detect_yolo(frame, model, device, imgsz=640, conf_thres=0.4, iou_thres=0.7):
    """YOLO 目标检测
    
    参数:
        frame: 输入的BGR图像
        model: YOLO模型
        device: 计算设备
        imgsz: 输入图像大小
        conf_thres: 置信度阈值
        iou_thres: IoU阈值
    
    返回:
        detections: 检测结果列表 [[x1, y1, x2, y2, conf, cls], ...]
    """
    if not TORCH_AVAILABLE or not YOLOV5_AVAILABLE:
        return []
    
    if model is None or frame is None:
        return []
    
    try:
        img0 = frame.copy()
        img, ratio, pad = letterbox(img0, new_shape=imgsz, auto=False)

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        
        # 根据模型类型选择精度
        if device.type != 'cpu':
            img = img.half()  # FP16
        else:
            img = img.float()  # FP32
        
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 禁用梯度计算
        with torch.no_grad():
            pred = model(img, augment=False)[0]
        
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

        detections = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape, ratio_pad=(ratio, pad)).round()
                for *xyxy, conf, cls in reversed(det.cpu().numpy()):
                    detections.append([*xyxy, conf, int(cls)])

        return detections
    except Exception as e:
        print(f"⚠️ YOLO检测错误: {e}")
        return []


def load_yolo_model(weights_path, device=None):
    """加载YOLO模型
    
    参数:
        weights_path: 权重文件路径
        device: 计算设备，如果为None则自动选择
    
    返回:
        (model, device): 模型和设备
    """
    if not TORCH_AVAILABLE:
        print("⚠️ torch 不可用，无法加载YOLO模型")
        return None, None
    
    if not YOLOV5_AVAILABLE:
        print("⚠️ YOLOv5 不可用，无法加载模型")
        return None, None
    
    try:
        # 初始化设备
        if device is None:
            set_logging()
            device = select_device('')
        
        # 加载模型
        model = attempt_load(weights_path, device=device)
        model.half() if device.type != 'cpu' else model.float()
        model.eval()
        
        # 预热模型
        if device.type != 'cpu':
            dummy_img = torch.zeros((1, 3, 640, 640), device=device).type_as(next(model.parameters()))
            model(dummy_img)
        
        print(f"✅ YOLO模型已加载: {weights_path}")
        print(f"   设备: {device}")
        if hasattr(model, 'names'):
            print(f"   类别: {model.names}")
        
        return model, device
    except Exception as e:
        print(f"❌ YOLO模型加载失败: {e}")
        return None, device if device is not None else None


def init_yolo(weights_path="yolov5/runs/train/my_custom_train8/weights/best.pt"):
    """初始化YOLO检测器的便捷函数
    
    参数:
        weights_path: 权重文件路径
    
    返回:
        (model, device): 模型和设备
    """
    return load_yolo_model(weights_path)
