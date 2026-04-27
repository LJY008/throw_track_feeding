# -*- coding: utf-8 -*-
"""
相机模块 - Orbbec相机和模拟相机数据源
从 neck_throat_track.py 分离出的相机相关代码
"""

import os
import sys
import time
import cv2
import numpy as np

# 获取脚本目录（父目录，因为neck_throat_v2是子目录）
script_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# 添加 openpose_bin 到搜索路径（pyorbbecsdk.pyd 在这里）
openpose_bin_dir = os.path.join(script_dir, 'openpose_bin')
if openpose_bin_dir not in sys.path:
    sys.path.insert(0, openpose_bin_dir)

# 添加 Orbbec 支持
ORBBEC_AVAILABLE = False
UTILS_IMPORTED = False
frame_to_bgr_image = None
OBFormat = None
OBFrameType = None

try:
    orbbec_path = os.path.join(script_dir, 'orbbec_examples')
    if orbbec_path not in sys.path:
        sys.path.insert(0, orbbec_path)
    
    from pyorbbecsdk import (Pipeline, Config, OBSensorType, OBFormat, 
                             AlignFilter, OBStreamType, OBPropertyID, 
                             OBAlignMode, OBPermissionType, OBPoint2f, 
                             Context, OBFrameType)
    import pyorbbecsdk as ob
    
    # 导入 Orbbec utils
    try:
        import importlib.util
        orbbec_utils_path = os.path.join(orbbec_path, 'utils.py')
        spec = importlib.util.spec_from_file_location("orbbec_utils", orbbec_utils_path)
        orbbec_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(orbbec_utils)
        frame_to_bgr_image = orbbec_utils.frame_to_bgr_image
        UTILS_IMPORTED = True
        print("✅ Orbbec工具函数已导入")
    except Exception as e:
        UTILS_IMPORTED = False
        frame_to_bgr_image = None
        print("⚠️ 无法导入Orbbec工具函数")

    ORBBEC_AVAILABLE = True
    print("✅ Orbbec SDK 可用")
except ImportError as e:
    ORBBEC_AVAILABLE = False
    UTILS_IMPORTED = False
    frame_to_bgr_image = None
    print("⚠️ Orbbec SDK 不可用")


class OrbbecCameraSource:
    """Orbbec相机数据源 - 支持彩色和深度图像"""
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.align_filter = None
        self.running = False
        self.depth_scale = 1.0
        self.depth_intrinsics = None
        self.color_intrinsics = None
        self.extrinsic = None
        self._frame_count = 0
        self._depth_format_warned = False
        self._depth_scale_printed = False
        self._depth_ok_printed = False
        self._depth_resize_warned = False
        self._depth_exc_printed = False
        self._no_depth_warned = False

    def initialize(self, timeout=5.0):
        """初始化Orbbec Femto相机"""
        if not ORBBEC_AVAILABLE:
            print("❌ Orbbec SDK 不可用")
            return False
            
        print(f"   检查已连接的设备...")
        try:
            context = Context()
            device_list = context.query_devices()
            device_count = device_list.get_count()
            
            if device_count < 1:
                print(f"❌ 未检测到Orbbec设备")
                return False
            
            print(f"✅ 检测到 {device_count} 个Orbbec设备")
        except Exception as e:
            print(f"❌ 设备枚举失败: {e}")
            return False
        
        try:
            self.pipeline = Pipeline()
            self.config = Config()
            device = self.pipeline.get_device()
        except Exception as e:
            print(f"❌ Pipeline初始化失败: {e}")
            return False
        
        try:
            device_info = device.get_device_info()
            device_name = device_info.get_name()
            print(f"📷 检测到设备: {device_name}")
            
            is_femto = "femto" in device_name.lower()
            if is_femto:
                print("✅ 检测到Femto系列相机")
            
            color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = None
            depth_profile = None
            hw_d2c_enabled = False
            
            if is_femto:
                try:
                    preferred_resolutions = [(640, 480), (1280, 720), (640, 400)]
                    for width, height in preferred_resolutions:
                        for i in range(len(color_profile_list)):
                            test_color_profile = color_profile_list[i]
                            if (test_color_profile.get_format() == OBFormat.RGB and 
                                test_color_profile.get_width() == width and 
                                test_color_profile.get_height() == height):
                                hw_d2c_profiles = self.pipeline.get_d2c_depth_profile_list(
                                    test_color_profile, OBAlignMode.HW_MODE)
                                if len(hw_d2c_profiles) > 0:
                                    for hw_profile in hw_d2c_profiles:
                                        if hw_profile.get_fps() == 30:
                                            depth_profile = hw_profile
                                            break
                                    if depth_profile is None:
                                        depth_profile = hw_d2c_profiles[0]
                                    color_profile = test_color_profile
                                    hw_d2c_enabled = True
                                    print(f"✅ 硬件D2C对齐: RGB {width}x{height}")
                                    break
                        if hw_d2c_enabled:
                            break
                except Exception as e:
                    print(f"⚠️ 硬件对齐配置失败: {e}")
            
            if not hw_d2c_enabled:
                try:
                    color_profile = color_profile_list.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
                except:
                    try:
                        color_profile = color_profile_list.get_video_stream_profile(640, 480, OBFormat.MJPG, 30)
                    except:
                        color_profile = color_profile_list.get_default_video_stream_profile()
                
                depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                try:
                    depth_profile = depth_profile_list.get_video_stream_profile(640, 576, OBFormat.Y16, 30)
                except:
                    depth_profile = depth_profile_list.get_default_video_stream_profile()
            
            if hw_d2c_enabled:
                self.config.enable_stream(depth_profile)
                self.config.enable_stream(color_profile)
                self.config.set_align_mode(OBAlignMode.HW_MODE)
                self.align_filter = None
            else:
                self.config.enable_stream(color_profile)
                self.config.enable_stream(depth_profile)

            if is_femto:
                try:
                    if device.is_property_supported(
                        OBPropertyID.OB_STRUCT_CURRENT_DEPTH_ALG_MODE,
                        OBPermissionType.PERMISSION_READ_WRITE):
                        work_mode_list = device.get_depth_work_mode_list()
                        if work_mode_list and work_mode_list.get_count() > 0:
                            mode = work_mode_list.get_depth_work_mode_by_index(0)
                            device.set_depth_work_mode(mode.name)
                except Exception as e:
                    pass

            self.pipeline.enable_frame_sync()
            
            if self.align_filter is None and not is_femto:
                self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

            self.pipeline.start(self.config)
            self.running = True
            print("✅ 相机流已启动")
            
            try:
                depth_profile_intrinsics = depth_profile.as_video_stream_profile()
                self.depth_intrinsics = depth_profile_intrinsics.get_intrinsic()
                color_profile_intrinsics = color_profile.as_video_stream_profile()
                self.color_intrinsics = color_profile_intrinsics.get_intrinsic()
                self.extrinsic = depth_profile.get_extrinsic_to(color_profile)
                print(f"✅ 相机内参已获取")
            except Exception as e:
                print(f"⚠️ 获取相机内参失败: {e}")

            for test_attempt in range(10):
                test_frames = self.pipeline.wait_for_frames(1000)
                if test_frames is not None:
                    test_color_frame = test_frames.get_color_frame()
                    test_depth_frame = test_frames.get_depth_frame()
                    if test_color_frame is not None and test_depth_frame is not None:
                        try:
                            self.depth_scale = test_depth_frame.get_value_scale()
                        except:
                            pass
                        print(f"✅ 相机测试成功")
                        return True
                time.sleep(0.3)

            return True

        except Exception as e:
            print(f"❌ Orbbec相机初始化失败: {e}")
            return False

    def get_frames(self):
        """获取彩色和深度图像帧"""
        if not self.running:
            return None, None

        try:
            frames = self.pipeline.wait_for_frames(50)
            if frames is None:
                return None, None

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame() if self.align_filter is None else frames.get_depth_frame()
            
            if depth_frame is None and OBFrameType is not None:
                try:
                    for i in range(frames.get_count()):
                        frame = frames.get_frame_by_index(i)
                        if frame and frame.get_type() == OBFrameType.DEPTH_FRAME:
                            depth_frame = frame
                            break
                except:
                    pass
            
            self._frame_count += 1
            
            if color_frame is None:
                return None, None

            color_image = None
            color_format = color_frame.get_format()
            
            if color_format == OBFormat.MJPG:
                data = np.asanyarray(color_frame.get_data())
                color_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            elif UTILS_IMPORTED and frame_to_bgr_image is not None:
                try:
                    color_image = frame_to_bgr_image(color_frame)
                except:
                    return None, None
            
            if color_image is None:
                return None, None

            depth_image = None
            if depth_frame is not None:
                try:
                    if depth_frame.get_format() == OBFormat.Y16:
                        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                            (depth_frame.get_height(), depth_frame.get_width()))
                        depth_image = depth_data.astype(np.uint16)
                        
                        width = color_frame.get_width()
                        height = color_frame.get_height()
                        if depth_image.shape[1] != width or depth_image.shape[0] != height:
                            depth_image = cv2.resize(depth_image, (width, height), interpolation=cv2.INTER_NEAREST)
                except Exception as e:
                    if not self._depth_exc_printed:
                        print(f"❌ 深度图像转换异常: {e}")
                        self._depth_exc_printed = True

            return color_image, depth_image

        except Exception as e:
            return None, None

    def release(self):
        """释放相机资源"""
        if self.running:
            self.pipeline.stop()
            self.running = False
            print("Orbbec相机已释放")


class SimulatedCameraSource:
    """模拟相机数据源"""
    def __init__(self):
        self.running = False
        self.frame_count = 0
        self.start_time = None
        self.width = 640
        self.height = 480
        self.depth_intrinsics = None
        self.color_intrinsics = None
        self.extrinsic = None
        print("📷 [模拟模式] 使用模拟相机数据源")
    
    def initialize(self):
        self.running = True
        self.start_time = time.time()
        print("✅ [模拟模式] 模拟相机已初始化")
        return True
    
    def get_frames(self):
        if not self.running:
            return None, None
        
        self.frame_count += 1
        current_time = time.time() - self.start_time
        
        color_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        neck_x, neck_y = self.width // 2, self.height // 2 - 50
        cv2.rectangle(color_image, (neck_x - 80, neck_y - 100), (neck_x + 80, neck_y + 100), (100, 100, 100), -1)
        cv2.putText(color_image, "Neck Area", (neck_x - 40, neck_y - 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        target_base_x = self.width // 2
        target_base_y = self.height // 2 + 30
        
        period = 6.0
        phase = (current_time % period) / period
        
        if phase < 0.15:
            y_offset = phase * 100
        elif phase < 0.3:
            y_offset = 15 + (phase - 0.15) * 50
        elif phase < 0.5:
            y_offset = 22.5 - (phase - 0.3) * 112.5
        else:
            y_offset = 0
        
        target_y = int(target_base_y + y_offset)
        target_x = target_base_x + int(10 * np.sin(current_time * 0.5))
        
        cv2.circle(color_image, (target_x, target_y), 15, (0, 255, 0), -1)
        cv2.putText(color_image, "Target", (target_x + 20, target_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(color_image, "Press 'q' to quit, SPACE for gold standard", (20, self.height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(color_image, "SIMULATION MODE", (20, self.height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        depth_image = np.ones((self.height, self.width), dtype=np.uint16) * 1000
        depth_image[neck_y-100:neck_y+100, neck_x-80:neck_x+80] = 800
        cv2.circle(depth_image, (target_x, target_y), 15, 750 + int(y_offset), -1)
        
        time.sleep(0.033)
        
        return color_image, depth_image
    
    def release(self):
        self.running = False
        print("✅ [模拟模式] 模拟相机已释放")
    
    def get_simulated_target_position(self):
        if self.start_time is None:
            return None, None, None
        
        current_time = time.time() - self.start_time
        period = 6.0
        phase = (current_time % period) / period
        
        if phase < 0.15:
            y_displacement = phase * 100 * 0.2
        elif phase < 0.3:
            y_displacement = 3 + (phase - 0.15) * 50 * 0.2
        elif phase < 0.5:
            y_displacement = 4.5 - (phase - 0.3) * 112.5 * 0.2
        else:
            y_displacement = 0
        
        z_displacement = y_displacement * 0.3 + np.random.normal(0, 0.5)
        y_displacement += np.random.normal(0, 0.3)
        
        return current_time, y_displacement, z_displacement


def get_depth_at_point(depth_image, x, y, window_size=7):
    """获取指定点的深度值（使用窗口均值滤波）"""
    if depth_image is None:
        return None
    
    h, w = depth_image.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return None
    
    half_window = window_size // 2
    y_min = max(0, y - half_window)
    y_max = min(h, y + half_window + 1)
    x_min = max(0, x - half_window)
    x_max = min(w, x + half_window + 1)
    
    window = depth_image[y_min:y_max, x_min:x_max]
    valid_depths = window[window > 0]
    
    if len(valid_depths) > 0:
        return np.mean(valid_depths)
    
    search_window = 20
    y_min2 = max(0, y - search_window)
    y_max2 = min(h, y + search_window + 1)
    x_min2 = max(0, x - search_window)
    x_max2 = min(w, x + search_window + 1)
    
    window2 = depth_image[y_min2:y_max2, x_min2:x_max2]
    valid_depths2 = window2[window2 > 0]
    
    if len(valid_depths2) > 0:
        return np.mean(valid_depths2)
    
    return None
