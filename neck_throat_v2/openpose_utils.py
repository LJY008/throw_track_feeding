# -*- coding: utf-8 -*-
"""
OpenPose工具模块 - 姿态检测相关函数
从 neck_throat_track.py 分离出的OpenPose相关代码
"""

import os
import sys
from sys import platform
import numpy as np

# 获取脚本目录（父目录）
script_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
openpose_bin_dir = os.path.join(script_dir, 'openpose_bin')

# 配置OpenPose路径
if openpose_bin_dir not in sys.path:
    sys.path.insert(0, openpose_bin_dir)

if platform == "win32":
    os.environ['PATH'] = openpose_bin_dir + ';' + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(openpose_bin_dir)

# 导入OpenPose
OPENPOSE_AVAILABLE = False
op = None
try:
    if platform == "win32":
        import pyopenpose as op
    else:
        sys.path.append('../../python')
        from openpose import pyopenpose as op
    OPENPOSE_AVAILABLE = True
    print("✅ OpenPose 已导入")
except ImportError as e:
    OPENPOSE_AVAILABLE = False
    op = None
    print('⚠️ OpenPose 不可用')


def get_neck_keypoints(pose_keypoints):
    """从姿态关键点中提取颈部相关信息"""
    if pose_keypoints is None or len(pose_keypoints.shape) < 3:
        return None

    neck_info = {}

    if pose_keypoints.shape[0] > 0:
        person_keypoints = pose_keypoints[0]

        # 颈部关键点 (索引1)
        if len(person_keypoints) > 1:
            neck = person_keypoints[1]
            if neck[2] > 0.1:
                neck_info['neck'] = (int(neck[0]), int(neck[1]), neck[2])

        # 右肩膀 (索引2)
        if len(person_keypoints) > 2:
            right_shoulder = person_keypoints[2]
            if right_shoulder[2] > 0.1:
                neck_info['right_shoulder'] = (int(right_shoulder[0]), int(right_shoulder[1]), right_shoulder[2])

        # 左肩膀 (索引5)
        if len(person_keypoints) > 5:
            left_shoulder = person_keypoints[5]
            if left_shoulder[2] > 0.1:
                neck_info['left_shoulder'] = (int(left_shoulder[0]), int(left_shoulder[1]), left_shoulder[2])

    return neck_info


def get_face_keypoints(face_keypoints):
    """从脸部关键点中提取下巴信息"""
    if face_keypoints is None or len(face_keypoints.shape) < 3:
        return None

    face_info = {}

    # 提取第一个检测到的人的脸部关键点
    if face_keypoints.shape[0] > 0:
        person_face_keypoints = face_keypoints[0]

        # 下巴中心 (索引8) - 这是颈部区域的上边界
        if len(person_face_keypoints) > 8:
            chin_center = person_face_keypoints[8]
            if chin_center[2] > 0.1:
                face_info['chin_center'] = (int(chin_center[0]), int(chin_center[1]), chin_center[2])

    return face_info


def get_neck_region_bbox(neck_info, face_info):
    """
    计算颈部区域的边界框（从下巴到颈部关键点）
    返回: (x1, y1, x2, y2) 或 None
    """
    if not neck_info or 'neck' not in neck_info:
        return None
    
    if not face_info or 'chin_center' not in face_info:
        return None
    
    neck_pos = neck_info['neck']
    chin_pos = face_info['chin_center']
    
    # 计算颈部宽度（基于肩膀宽度）
    if 'left_shoulder' in neck_info and 'right_shoulder' in neck_info:
        left_shoulder_x = neck_info['left_shoulder'][0]
        right_shoulder_x = neck_info['right_shoulder'][0]
        neck_width = abs(left_shoulder_x - right_shoulder_x) // 3  # 颈部宽度约为肩宽的1/3
    else:
        neck_width = 60  # 默认颈部宽度

    # 计算矩形边界（从下巴到颈部）
    top_y = min(chin_pos[1], neck_pos[1]) - 5  # 下巴位置（稍微扩展）
    bottom_y = max(chin_pos[1], neck_pos[1]) + 10  # 颈部位置（稍微扩展）
    center_x = (neck_pos[0] + chin_pos[0]) // 2
    left_x = center_x - neck_width // 2
    right_x = center_x + neck_width // 2
    
    return (left_x, top_y, right_x, bottom_y)


def init_openpose(model_folder="./openpose_model/", net_resolution="256x256", 
                  number_people_max=1):
    """初始化OpenPose
    
    参数:
        model_folder: OpenPose模型文件夹路径
        net_resolution: 网络输入分辨率
        number_people_max: 最大检测人数
    
    返回:
        opWrapper: OpenPose包装器对象，如果初始化失败则返回None
    """
    if not OPENPOSE_AVAILABLE:
        print("❌ OpenPose 不可用，无法初始化")
        return None
    
    try:
        params = {
            "model_folder": model_folder,
            "net_resolution": net_resolution,
            "model_pose": "BODY_25",
            "face": True,
            "face_detector": 1,
            "number_people_max": number_people_max,
            "render_pose": 1,
            "disable_blending": True
        }
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        print("✅ OpenPose 初始化完成")
        return opWrapper
    except Exception as e:
        print(f"❌ OpenPose 初始化失败: {e}")
        return None


def process_frame(opWrapper, frame):
    """处理单帧图像，返回检测结果
    
    参数:
        opWrapper: OpenPose包装器对象
        frame: 输入的BGR图像
    
    返回:
        (neck_info, face_info, neck_region_bbox, output_image)
    """
    if opWrapper is None or frame is None:
        return None, None, None, None
    
    try:
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        
        neck_info = get_neck_keypoints(datum.poseKeypoints)
        face_info = get_face_keypoints(datum.faceKeypoints)
        neck_region_bbox = get_neck_region_bbox(neck_info, face_info)
        output_img = datum.cvOutputData.copy() if datum.cvOutputData is not None else None
        
        return neck_info, face_info, neck_region_bbox, output_img
    except Exception as e:
        print(f"⚠️ OpenPose处理帧失败: {e}")
        return None, None, None, None
