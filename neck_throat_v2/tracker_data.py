# -*- coding: utf-8 -*-
"""
追踪数据管理模块 - TrackerData类
从 neck_throat_track.py 分离出的数据追踪相关代码
"""

import os
import time
import csv
import cv2
import numpy as np
from collections import deque
import math


class TrackerData:
    """追踪数据管理（单目标模式 - 只有一个圆片）"""
    def __init__(self, window_size=300, display_window=60):
        # 单目标模式：使用deque优化性能（避免list.pop(0)的O(n)开销）
        # 预估最大容量：30fps * 300s = 9000点
        max_points = int(30 * window_size * 1.2)  # 留20%余量
        self.positions = deque(maxlen=max_points)  # 自动丢弃超出部分
        self.timestamps = deque(maxlen=max_points)
        self.neck_origin = None  # 颈部原点 (x0, y0, depth0)
        self.window_size = window_size
        self.target_track_id = None  # 当前追踪的ID（仅用于显示）
        self.last_update_time = None  # 最后一次更新时间
        self.id_timeout = 2.0  # ID超时时间
        
        # ========== 全量数据存储（用于最终导出和分析） ==========
        # 这些容器不限制大小，保存采集期间的所有数据
        self.full_positions = []  # 全量位置数据 [(y_disp, z_disp), ...]
        self.full_timestamps = []  # 全量时间戳
        
        # 展示用窗口大小（秒），用于UI实时绘图
        self.display_window = display_window
        
        # 相机内参（用于像素到3D坐标转换）
        self.depth_intrinsics = None  # 深度相机内参
        self.color_intrinsics = None  # 彩色相机内参
        self.extrinsic = None  # 深度到彩色的外参矩阵
        
        # 官方时间域滤波
        self.filtered_depth_value = None  # 当前滤波后的深度值
        self.depth_smoothing_alpha = 0.5  # 平滑系数
        
        # 颈部原点位置滤波（使用deque固定大小）
        self.neck_buffer_size = 15
        self.neck_position_buffer_x = deque(maxlen=self.neck_buffer_size)
        self.neck_position_buffer_y = deque(maxlen=self.neck_buffer_size)
        self.neck_position_buffer_depth = deque(maxlen=self.neck_buffer_size)
        self.filtered_neck_x = None
        self.filtered_neck_y = None
        self.filtered_neck_depth = None
        self.neck_smoothing_alpha = 0.3
        self.neck_filter_enabled = True
        
        # 双阈值滤波参数
        self.bilateral_filter_enabled = True
        self.bilateral_small_threshold_y = 2.0
        self.bilateral_large_threshold_y = 50.0
        self.bilateral_small_threshold_depth = 3.0
        self.bilateral_large_threshold_depth = 80.0
        
        # 颈喉连线斜率监测（身体运动检测）- 使用deque固定大小
        self.line_slope_window = 60
        self.line_slope_history = deque(maxlen=self.line_slope_window)
        self.slope_stable_threshold = 0.15
        self.body_tilt_detection_enabled = True  # 是否启用基于人体框架倾斜度的检测开关
        self.body_motion_detected = False
        self.body_motion_grace_frames = 30
        self.body_motion_cooldown = 0
        self.baseline_adapt_alpha = 0.001
        
        # QV波动率检测（使用deque固定大小）
        self.qv_enabled = False
        self.qv_window_duration = 1.0
        self.qv_z_history = deque(maxlen=100)  # 约3秒数据 @ 30fps
        self.qv_kernel_bandwidth = 0.5
        self.qv_volatility_threshold = 1.5
        self.qv_low_volatility_threshold = 0.3
        self.current_volatility = 0.0
        self.qv_suppress_slow_motion = True
    
    def pixel_to_3d(self, x, y, depth):
        """将彩色图像上的像素坐标(x, y)和深度值转换为3D坐标(X, Y, Z)单位mm"""
        if self.color_intrinsics is None or depth <= 0:
            return None
        
        try:
            fx = self.color_intrinsics.fx
            fy = self.color_intrinsics.fy
            cx = self.color_intrinsics.cx
            cy = self.color_intrinsics.cy
            
            X = (float(x) - cx) * depth / fx
            Y = (float(y) - cy) * depth / fy
            Z = depth
            
            return (X, Y, Z)
        except Exception as e:
            print(f"⚠️ 坐标转换失败: {e}")
            return None

    def set_neck_origin(self, x, y, depth):
        """设置颈部原点 - 使用滑动窗口中值滤波 + EMA"""
        if depth is not None and depth > 0:
            # 添加到历史缓冲区（deque自动限制大小）
            self.neck_position_buffer_x.append(x)
            self.neck_position_buffer_y.append(y)
            self.neck_position_buffer_depth.append(depth)
            
            # 计算中值（deque转list一次，避免多次转换）
            buf_x = list(self.neck_position_buffer_x)
            buf_y = list(self.neck_position_buffer_y)
            buf_depth = list(self.neck_position_buffer_depth)
            median_x = np.median(buf_x)
            median_y = np.median(buf_y)
            median_depth = np.median(buf_depth)
            
            # 首次设置或应用EMA滤波
            if self.filtered_neck_x is None:
                self.filtered_neck_x = float(median_x)
                self.filtered_neck_y = float(median_y)
                self.filtered_neck_depth = float(median_depth)
                print("✅ 初始化颈部原点: X={:.1f}, Y={:.1f}, Depth={:.1f}mm".format(
                    self.filtered_neck_x, self.filtered_neck_y, self.filtered_neck_depth))
            else:
                alpha = self.neck_smoothing_alpha
                self.filtered_neck_x = alpha * median_x + (1 - alpha) * self.filtered_neck_x
                self.filtered_neck_y = alpha * median_y + (1 - alpha) * self.filtered_neck_y
                self.filtered_neck_depth = alpha * median_depth + (1 - alpha) * self.filtered_neck_depth
            
            # 更新neck_origin为滤波后的值
            self.neck_origin = (int(self.filtered_neck_x), int(self.filtered_neck_y), self.filtered_neck_depth)

    def _calculate_line_slopes(self, neck_3d, throat_3d):
        """计算颈部到喉部连线在两个平面的斜率"""
        import math
        
        dx = throat_3d[0] - neck_3d[0]
        dy = throat_3d[1] - neck_3d[1]
        dz = throat_3d[2] - neck_3d[2]
        
        slope_yz = math.atan2(dy, dz) if dz != 0 else 0.0
        slope_xz = math.atan2(dx, dz) if dz != 0 else 0.0
        
        return slope_yz, slope_xz
    
    def _qv_kernel(self, u):
        """QV核函数"""
        if abs(u) <= 1.0:
            return (15.0 / 4.0) * (u - u**3)
        return 0.0
    
    def _calculate_qv_volatility(self, current_time):
        """计算QV波动率（优化版：使用deque和向量化计算）"""
        if not self.qv_enabled or len(self.qv_z_history) < 3:
            return 0.0
        
        # deque自动限制大小，这里只需过滤时间窗口
        # 转换为list进行处理（deque不支持切片赋值）
        history_list = [(t, z) for t, z in self.qv_z_history 
                        if current_time - t <= self.qv_window_duration]
        
        if len(history_list) < 3:
            return 0.0
        
        # 向量化计算
        times = np.array([t for t, _ in history_list])
        z_values = np.array([z for _, z in history_list])
        z_diff = np.diff(z_values)
        qv_increments = z_diff ** 2
        
        h = self.qv_kernel_bandwidth
        
        # 向量化权重计算
        t_array = times[1:]  # 对应z_diff的时间点
        u_array = (current_time - t_array) / h
        # 向量化核函数
        weights = np.where(np.abs(u_array) <= 1.0, 
                          (15.0 / 4.0) * (u_array - u_array**3), 
                          0.0)
        
        weight_sum = np.sum(weights)
        
        if weight_sum < 1e-6:
            return 0.0
        
        weighted_qv = np.sum(weights * qv_increments) / weight_sum
        
        if len(self.qv_z_history) >= 2:
            dt = self.qv_z_history[-1][0] - self.qv_z_history[-2][0]
            if dt > 0:
                volatility = np.sqrt(abs(weighted_qv) / dt)
            else:
                volatility = 0.0
        else:
            volatility = 0.0
        
        return volatility
    
    def _is_slow_body_motion_qv(self, current_time):
        """基于QV波动率判断是否为缓慢的身体移动"""
        if not self.qv_enabled or not self.qv_suppress_slow_motion:
            return False
        
        volatility = self._calculate_qv_volatility(current_time)
        self.current_volatility = volatility
        
        if volatility < self.qv_low_volatility_threshold:
            return True
        
        return False

    def add_position(self, track_id, x, y, depth, timestamp):
        """添加位置 - 使用真实3D坐标差"""
        if self.neck_origin is None:
            return
        
        # 单目标模式：只记录当前ID用于显示
        if self.target_track_id != track_id:
            if self.target_track_id is not None:
                print(f"📌 ID变化: {self.target_track_id} → {track_id}")
            self.target_track_id = track_id
        
        self.last_update_time = timestamp
        
        # 初始化深度滤波值
        if self.filtered_depth_value is None and depth is not None and depth > 0:
            self.filtered_depth_value = depth
        
        # 深度值处理
        if depth is not None and depth > 0:
            if self.filtered_depth_value is None:
                self.filtered_depth_value = float(depth)
            else:
                self.filtered_depth_value = cv2.addWeighted(
                    np.array([[depth]], dtype=np.float32), 
                    self.depth_smoothing_alpha,
                    np.array([[self.filtered_depth_value]], dtype=np.float32), 
                    1 - self.depth_smoothing_alpha, 
                    0
                )[0, 0]
        
        depth_value = self.filtered_depth_value if self.filtered_depth_value is not None else 0
        
        # 没有彩色相机内参时使用简化坐标
        if self.color_intrinsics is None:
            y_rel = (y - self.neck_origin[1]) * 0.5
            depth_rel = (depth_value - self.neck_origin[2])
            
            if not hasattr(self, 'throat_baseline_y'):
                self.throat_baseline_y = y_rel
                self.throat_baseline_z = depth_rel
                swallow_motion_y = 0.0
                swallow_motion_z = 0.0
            else:
                swallow_motion_y = y_rel - self.throat_baseline_y
                swallow_motion_z = depth_rel - self.throat_baseline_z
                self.throat_baseline_y += self.baseline_adapt_alpha * swallow_motion_y
                self.throat_baseline_z += self.baseline_adapt_alpha * swallow_motion_z
            
            self.positions.append((swallow_motion_y, swallow_motion_z))
            self.timestamps.append(timestamp)
            
            # 同时保存到全量数据容器（用于最终导出）
            self.full_positions.append((swallow_motion_y, swallow_motion_z))
            self.full_timestamps.append(timestamp)
            
            # deque自动限制大小，无需手动pop
            
            return
        
        # 3D坐标转换
        throat_3d = self.pixel_to_3d(x, y, depth_value)
        neck_3d = self.pixel_to_3d(
            self.neck_origin[0], 
            self.neck_origin[1], 
            self.neck_origin[2]
        )
        
        if throat_3d is None or neck_3d is None:
            return
        
        # 初始化基线
        if not hasattr(self, 'throat_baseline_y'):
            self.throat_baseline_x = throat_3d[0]
            self.throat_baseline_y = throat_3d[1]
            self.throat_baseline_z = throat_3d[2]
            swallow_motion_y = 0.0
            swallow_motion_z = 0.0
        else:
            current_time = time.time()
            swallow_motion_y = throat_3d[1] - self.throat_baseline_y
            swallow_motion_z = throat_3d[2] - self.throat_baseline_z
            
            # 【开关】基于人体框架倾斜度的检测（仅在启用时执行）
            if self.body_tilt_detection_enabled:
                # 斜率检测
                slope_yz, slope_xz = self._calculate_line_slopes(neck_3d, throat_3d)
                
                # deque自动限制大小，无需手动裁剪
                self.line_slope_history.append((current_time, slope_yz, slope_xz))
                
                # 只需要在计算时过滤时间窗口内的数据
                window_duration = self.line_slope_window / 30.0
                recent_slopes = [(t, syz, sxz) for t, syz, sxz in self.line_slope_history 
                                 if current_time - t <= window_duration]
                
                body_motion_now = False
                if len(recent_slopes) >= 2:
                    slopes_yz = [s[1] for s in recent_slopes]
                    slopes_xz = [s[2] for s in recent_slopes]
                    
                    slope_yz_range = max(slopes_yz) - min(slopes_yz)
                    slope_xz_range = max(slopes_xz) - min(slopes_xz)
                    
                    if slope_yz_range > self.slope_stable_threshold or slope_xz_range > self.slope_stable_threshold:
                        body_motion_now = True
                        self.body_motion_cooldown = self.body_motion_grace_frames
                
                if self.body_motion_cooldown > 0:
                    self.body_motion_cooldown -= 1
                    body_motion_now = True
                    
                self.body_motion_detected = body_motion_now
            else:
                # 当开关禁用时，始终将body_motion_detected设为False
                self.body_motion_detected = False
            
            # 基线自适应
            self.throat_baseline_x = self.baseline_adapt_alpha * throat_3d[0] + (1 - self.baseline_adapt_alpha) * self.throat_baseline_x
            self.throat_baseline_y = self.baseline_adapt_alpha * throat_3d[1] + (1 - self.baseline_adapt_alpha) * self.throat_baseline_y
            self.throat_baseline_z = self.baseline_adapt_alpha * throat_3d[2] + (1 - self.baseline_adapt_alpha) * self.throat_baseline_z
            
            # QV波动率检测
            if self.qv_enabled:
                self.qv_z_history.append((current_time, throat_3d[2]))
                is_slow_motion = self._is_slow_body_motion_qv(current_time)
                if is_slow_motion:
                    self.body_motion_detected = True
        
        # 双边滤波
        def bilateral_filter(value, small_threshold, large_threshold):
            if abs(value) < small_threshold:
                return 0
            elif abs(value) > large_threshold:
                return large_threshold if value > 0 else -large_threshold
            else:
                return value
        
        if self.bilateral_filter_enabled:
            filtered_motion_y = bilateral_filter(swallow_motion_y, 
                self.bilateral_small_threshold_y, self.bilateral_large_threshold_y)
            filtered_motion_z = bilateral_filter(swallow_motion_z, 
                self.bilateral_small_threshold_depth, self.bilateral_large_threshold_depth)
        else:
            filtered_motion_y = swallow_motion_y
            filtered_motion_z = swallow_motion_z
        
        final_y = filtered_motion_y
        final_z = filtered_motion_z
        
        # 身体运动时置零
        if self.body_motion_detected:
            final_y = 0.0
            final_z = 0.0
        
        self.positions.append((final_y, final_z))
        self.timestamps.append(timestamp)
        
        # 同时保存到全量数据容器（用于最终导出）
        self.full_positions.append((final_y, final_z))
        self.full_timestamps.append(timestamp)
        
        # deque自动限制大小，无需手动pop

    def get_displacements(self):
        """获取位移数据"""
        if len(self.positions) < 2:
            return [], []
        
        # 直接从deque构建列表
        y_disps = [pos[0] for pos in self.positions]
        depth_values = [pos[1] for pos in self.positions]
        
        return y_disps, depth_values
    
    def get_full_displacements(self):
        """获取全量位移数据（用于最终导出和分析）"""
        if len(self.full_positions) < 2:
            return [], []
        
        y_disps = [pos[0] for pos in self.full_positions]
        depth_values = [pos[1] for pos in self.full_positions]
        
        return y_disps, depth_values
    
    def get_full_timestamps(self):
        """获取全量时间戳（用于最终导出和分析）"""
        return list(self.full_timestamps)
    
    def reset(self):
        """重置追踪器"""
        # 使用deque的clear()方法
        self.positions.clear()
        self.timestamps.clear()
        self.neck_origin = None
        self.target_track_id = None
        self.filtered_depth_value = None
        self.neck_position_buffer_x.clear()
        self.neck_position_buffer_y.clear()
        self.neck_position_buffer_depth.clear()
        self.filtered_neck_x = None
        self.filtered_neck_y = None
        self.filtered_neck_depth = None
        
        # 清空全量数据容器
        self.full_positions.clear()
        self.full_timestamps.clear()
        
        if hasattr(self, 'throat_baseline_y'):
            delattr(self, 'throat_baseline_y')
        if hasattr(self, 'throat_baseline_z'):
            delattr(self, 'throat_baseline_z')
        if hasattr(self, 'throat_baseline_x'):
            delattr(self, 'throat_baseline_x')
            
        self.line_slope_history.clear()
        self.qv_z_history.clear()
        self.body_motion_detected = False
        self.body_motion_cooldown = 0
        print("✅ 追踪器已重置")

    def get_timestamps(self):
        """获取时间戳"""
        return list(self.timestamps)
    
    def export_to_csv(self, filename="tracking_data.csv"):
        """导出追踪数据到CSV文件"""
        if len(self.positions) == 0:
            print("❌ 没有可导出的数据")
            return False
        # Ensure results directory exists and place file there if no path provided
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        try:
            os.makedirs(results_dir, exist_ok=True)
        except Exception:
            pass

        # If filename is just a name (no dir), save into results_dir
        if not os.path.dirname(filename):
            filename = os.path.join(results_dir, filename)

        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamps', 'y_displacement', 'z_displacement', 'displacements'])

                for i, (pos, ts) in enumerate(zip(self.positions, self.timestamps)):
                    y_disp = pos[0]
                    z_disp = pos[1]
                    writer.writerow([ts, y_disp, z_disp, z_disp])

            print(f"✅ Data exported to: {filename}")
            print(f"   Data points: {len(self.positions)}")
            if len(self.timestamps) >= 2:
                print(f"   Duration: {self.timestamps[-1] - self.timestamps[0]:.2f} s")
            return True

        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False
