# -*- coding: utf-8 -*-
"""
QV检测算法核心模块
包含 QVDetectionResult 数据类和 QVRealtimeDetector 检测器类
从 qv_realtime_monitor.py 分离，便于复用和测试
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class QVDetectionResult:
    """QV检测结果"""
    events: List[int]  # 事件索引
    event_times: List[float]  # 事件时间戳
    volatility: np.ndarray  # 波动率序列
    qv: np.ndarray  # 二次变分序列
    signal: np.ndarray  # 原始信号
    event_intervals: Optional[List[Tuple[float, float]]] = None  # 最佳事件区间（与events一一对应）
    event_candidates: Optional[List[List[int]]] = None  # 所有候选组
    all_event_intervals: Optional[List[Tuple[float, float]]] = None  # 所有组的区间（与event_candidates一一对应）
    best_event_group_indices: Optional[List[int]] = None  # 哪些组有最佳事件（组索引列表）
    acceleration: Optional[np.ndarray] = None  # 加速度数据


class QVRealtimeDetector:
    """QV实时检测器（基于QV_RT_OUR论文）"""
    
    def __init__(self, fs=30, h=0.5, H=1.5, H2=-2.0, suppression_window=1.0,
                 k0=5, min_displacement_change=1.0, window_size=1,
                 kalman_Q=[0.1, 1.0, 9.0], kalman_R_factor=0.1,
                 w1=0.7, w2=0.1, w3=0.2,
                 min_std=0.3, min_total_var=0.5,
                 suppression_enabled=True, verbose=False, **kwargs):
        """
        初始化QV检测器
        
        参数:
            fs: 采样频率 (Hz)
            h: 核平滑带宽 (秒) - 用于核函数加权
            H: 高波动率阈值 - 吞咽检测主阈值
            H2: 二阶导数阈值 - 防止误检
            suppression_window: 抑制窗口时长(秒) - 防止重复检测
            k0: 初始子采样步长 - QV计算参数
            min_displacement_change: 最小位移变化阈值(像素) - 验证真实运动
            window_size: 验证窗口大小 - 事件前后检查范围
            kalman_Q: 卡尔曼过程噪声协方差 [位置,速度,加速度]
            kalman_R_factor: 卡尔曼观测噪声因子
            suppression_enabled: 是否启用时间窗口抑制（在窗口内合并事件）
        """
        self.fs = fs
        self.dt = 1.0 / fs
        self.h = h
        self.H = H
        self.H2 = H2
        self.suppression_window = suppression_window
        self.suppression_enabled = suppression_enabled
        # 兼容老接口：接受但忽略 `verbose` 和其他多余关键字参数
        self.verbose = bool(verbose)
        self.k0 = k0
        self.min_displacement_change = min_displacement_change
        self.window_size = window_size
        self.kalman_Q = kalman_Q
        self.kalman_R_factor = kalman_R_factor
        # 双轴置零阈值（当任一轴绝对值小于该阈值时视为无效）
        self.zero_threshold = 1e-10
        # 组内最优选择的评分权重（用于_score_candidates）
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        # 位移窗口验证参数
        self.min_std = min_std
        self.min_total_var = min_total_var

        # 实时检测状态
        self.last_detection_time = -float('inf')
        self.current_time = 0.0
        
        # 创建最优核函数
        self.kernel = self._create_kernel()
    
    def _create_kernel(self):
        """创建最优一阶导估计核函数 (论文式9)"""
        def K(u):
            return (15.0 / 4.0) * (u - u ** 3) * (np.abs(u) <= 1.0)
        return K
    
    def _displacement_to_acceleration_kalman(self, displacement, timestamps):
        """
        使用卡尔曼滤波从位移数据计算加速度（与QV_RT_OUR.py一致）
        状态: [position, velocity, acceleration]
        
        参数:
            displacement: 位移数据
            timestamps: 时间戳
        
        返回:
            accelerations: 加速度数据
        """
        n = len(displacement)
        
        if n < 3:
            raise ValueError("Need at least 3 data points to calculate acceleration")
        
        # 初始化卡尔曼滤波器
        x = np.array([displacement[0], 0.0, 0.0], dtype=float)
        P = np.array([[1000.0, 0.0, 0.0],
                      [0.0, 1000.0, 0.0],
                      [0.0, 0.0, 1000.0]], dtype=float)
        
        # 观测噪声
        R = np.var(np.diff(displacement)) * self.kalman_R_factor
        if R <= 0:
            R = 1.0
        
        # 过程噪声
        Q = np.diag(self.kalman_Q)
        
        # 观测矩阵
        H_mat = np.array([[1.0, 0.0, 0.0]])
        
        # 存储结果
        accelerations = np.zeros(n)
        accelerations[0] = x[2]
        
        for i in range(1, n):
            dt = timestamps[i] - timestamps[i-1]
            
            # 状态转移矩阵
            F = np.array([
                [1.0, dt, 0.5*dt**2],
                [0.0, 1.0, dt],
                [0.0, 0.0, 1.0]
            ])
            
            # 预测
            x = F @ x
            P = F @ P @ F.T + Q
            
            # 更新
            z = np.array([displacement[i]], dtype=float)
            y = z - H_mat @ x
            S = H_mat @ P @ H_mat.T + R
            K = P @ H_mat.T / S[0, 0]
            
            x = x + K.flatten() * y[0]
            P = P - np.outer(K, H_mat @ P)
            
            accelerations[i] = x[2]
        
        return accelerations
    
    def calculate_qv(self, signal, k0=None):
        """
        计算二次变分 (QV) 使用子采样平均法
        
        参数:
            signal: 输入信号序列
            k0: 初始子采样步长（None则使用self.k0）
        
        返回:
            qv_avg: 平均QV序列
            k_star: 最优子采样步长
        """
        if k0 is None:
            k0 = self.k0
        n = len(signal)
        
        if n < 2:
            return np.zeros(n), 1
        
        # 处理NaN和无穷大值
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 1. 计算初始QV估计
        signal_diff = np.diff(signal)
        qv_initial = np.cumsum(signal_diff ** 2)
        
        # 确保k0在合理范围
        k0 = min(k0, n // 2)
        k0 = max(k0, 1)
        
        # 2. 计算最优子采样步长k*
        try:
            qv_mean = qv_initial[-1] / n if n > 0 else 0
            numerator = 9 * k0 * (qv_mean ** 2)
            
            if n <= k0:
                k_star = 1
            else:
                start_indices = np.arange(0, n - k0, k0)
                end_indices = start_indices + k0
                
                if len(start_indices) < 2:
                    k_star = 1
                else:
                    diff_signal = signal[end_indices] - signal[start_indices]
                    fourth_moment = np.sum(diff_signal ** 4)
                    
                    if fourth_moment <= 1e-10:
                        k_star = k0
                    else:
                        denominator = fourth_moment
                        ratio = numerator / denominator
                        
                        if ratio <= 0 or np.isnan(ratio) or np.isinf(ratio):
                            k_star = k0
                        else:
                            k_star = int(round(ratio ** (1 / 3)))
        except Exception:
            k_star = k0
        
        k_star = max(k_star, 1)
        k_star = min(k_star, n // 2)
        
        # 3. 使用k*创建子采样序列
        qv_subsampled = np.zeros(n)
        count = np.zeros(n)
        
        for j in range(k_star):
            subsampled = signal[j::k_star]
            
            if len(subsampled) < 2:
                continue
            
            subsampled_diff = np.diff(subsampled)
            subsampled_qv = np.cumsum(subsampled_diff ** 2)
            
            indices = np.arange(j + k_star, j + k_star + len(subsampled_qv) * k_star, k_star)
            valid_idx = indices < n
            
            if np.any(valid_idx):
                qv_subsampled[indices[valid_idx]] += subsampled_qv[:sum(valid_idx)]
                count[indices[valid_idx]] += 1
        
        count[count == 0] = 1
        
        # 4. 计算平均QV
        qv_avg = qv_subsampled / count
        return qv_avg, k_star
    
    def estimate_volatility(self, qv):
        """
        估计波动率 σ_t (论文式9)
        
        参数:
            qv: 二次变分序列
        
        返回:
            sigma_t: 波动率估计
            sigma_t2: 二阶导数
        """
        n = len(qv)
        sigma_t = np.zeros(n)
        sigma_t2 = np.zeros(n)
        
        # 计算核函数窗口大小
        m = int(2 * self.h / self.dt)
        m = max(m, 1)
        
        for t in range(n):
            # 计算当前点的核函数权重
            indices = np.arange(max(0, t - m), min(n, t + m + 1))
            u = (indices * self.dt - t * self.dt) / self.h
            weights = self.kernel(u)
            
            # 1. 估计波动率 σ_t
            qv_diff = np.diff(qv, prepend=0)[indices]
            weighted_sum = np.sum(weights * qv_diff)
            sigma_t[t] = np.sqrt(np.abs(weighted_sum) / (len(indices) * self.h))
            
            # 2. 估计二阶导数 σ_t''
            sigma_window = sigma_t[max(0, t - m):min(n, t + m + 1)]
            if len(sigma_window) == len(weights):
                sigma_t2[t] = np.sum(weights * sigma_window) / (len(indices) * self.h ** 2)
        
        return sigma_t, sigma_t2
    
    def _calc_disp_features(self, window):
        """计算位移窗口特征"""
        change = float(np.max(window) - np.min(window))
        std = float(np.std(window))
        total_var = float(np.sum(np.abs(np.diff(window))))
        return change, std, total_var
    
    def _is_valid_displacement_window(self, window, min_change=None, min_std=None, min_total_var=None):
        """验证位移窗口是否有效"""
        if min_change is None:
            min_change = self.min_displacement_change
        if min_std is None:
            min_std = self.min_std
        if min_total_var is None:
            min_total_var = self.min_total_var
        
        change, std, total_var = self._calc_disp_features(window)
        if np.all(np.abs(window) < 1e-6):
            return False
        return (change >= min_change and std >= min_std and total_var >= min_total_var)
    
    def _score_candidates(self, valid_indices, sigma_ap, sigma2_ap, displacement):
        """对候选事件打分并返回最佳索引"""
        if not valid_indices:
            return -1
        
        # 使用实例配置的权重（可在GUI中调整）
        w1, w2, w3 = self.w1, self.w2, self.w3
        sigma_vals = np.array([sigma_ap[i] for i in valid_indices])
        sigma2_vals = np.array([sigma2_ap[i] for i in valid_indices])
        disp_changes = []
        for i in valid_indices:
            win_start = max(0, i - self.window_size)
            win_end = min(len(displacement), i + self.window_size + 1)
            win = displacement[win_start:win_end]
            disp_changes.append(np.max(win) - np.min(win))
        disp_changes = np.array(disp_changes)
        
        sigma_max = np.max(sigma_vals) if len(sigma_vals) else 1
        sigma2_min = np.min(sigma2_vals) if len(sigma2_vals) else -1
        disp_max = np.max(disp_changes) if len(disp_changes) else 1
        
        scores = (w1 * (sigma_vals / (sigma_max + 1e-6)) +
                  w2 * (1 - np.abs(sigma2_vals) / (np.abs(sigma2_min) + 1e-6)) +
                  w3 * (disp_changes / (disp_max + 1e-6)))
        return valid_indices[int(np.argmax(scores))]
    
    def detect_events(self, sigma, sigma2):
        """
        检测事件（吞咽）- 简单版本
        
        参数:
            sigma: 波动率序列
            sigma2: 二阶导数序列
        
        返回:
            events: 事件索引列表
        """
        events = []
        
        for t in range(1, len(sigma) - 1):
            # 条件1: 局部极大值
            cond1 = (sigma[t - 1] <= sigma[t]) and (sigma[t] > sigma[t + 1])
            
            # 条件2: 波动率超过阈值
            cond2 = sigma[t] >= self.H
            
            # 条件3: 二阶导数低于阈值
            cond3 = sigma2[t] <= self.H2
            
            if cond1 and cond2 and cond3:
                events.append(t)
        
        return events
    
    def detect_swallows_groupwise_best(self, sigma_ap, sigma2_ap, timestamps, displacement_data, silent=False):
        """
        组内最优检测（与QV_RT_OUR.py一致）
        在抑制窗口内选择最佳候选事件
        
        时间窗口逻辑：
        - 启用抑制时：使用滑动窗口，窗口从最后一个候选事件开始计算
        - 持续时间：以组内第一个候选到最后一个候选的时间范围为准
        
        参数:
            silent: 是否静默模式（不打印日志）
        """
        from typing import List, Tuple, Optional
        
        detected_events: List[int] = []
        volatility_buffer: List[float] = []
        event_group: List[int] = []
        event_intervals: List[Tuple[float, float]] = []  # 最佳事件的区间
        all_event_intervals: List[Tuple[float, float]] = []  # 所有组的区间
        best_event_group_indices: List[int] = []  # 记录哪些组有最佳事件
        event_group_candidates: List[List[int]] = []
        last_candidate_time: Optional[float] = None  # 改为记录最后一个候选的时间（滑动窗口）
        
        if not silent:
            print(f"组内最优检测 | 抑制窗口={self.suppression_window}s | H={self.H} H2={self.H2}")
        
        def get_group_duration() -> Tuple[float, float]:
            """获取组内第一个候选到最后一个候选的时间范围"""
            if not event_group:
                return (0.0, 0.0)
            first_idx = event_group[0]
            last_idx = event_group[-1]
            
            # 向前扩展：找到第一个候选之前波动率开始上升的点
            start_idx = first_idx
            threshold = self.H * 0.3
            for j in range(first_idx, -1, -1):
                if sigma_ap[j] < threshold:
                    start_idx = j
                    break
                start_idx = j
            
            # 向后扩展：找到最后一个候选之后波动率回落的点
            end_idx = last_idx
            for j in range(last_idx, len(sigma_ap)):
                if sigma_ap[j] < threshold:
                    end_idx = j
                    break
                end_idx = j
            
            return timestamps[start_idx], timestamps[end_idx]
        
        def process_group(final: bool = False):
            nonlocal event_group, last_candidate_time
            if not event_group:
                return
            
            current_group_index = len(event_group_candidates)  # 当前组的索引
            event_group_candidates.append(event_group.copy())
            valid_events: List[int] = []
            
            for idx in event_group:
                win_start = max(0, idx - self.window_size)
                win_end = min(len(displacement_data), idx + self.window_size + 1)
                win = displacement_data[win_start:win_end]
                if self._is_valid_displacement_window(win):
                    valid_events.append(idx)
                elif not silent:
                    print(f"    过滤候选 @ {timestamps[idx]:.3f}s (位移变化不足)")
            
            # 使用组内第一个到最后一个候选的扩展时间范围
            event_start, event_end = get_group_duration()
            duration = event_end - event_start
            interval = (event_start, event_end)
            
            # 始终记录所有组的区间
            all_event_intervals.append(interval)
            
            if valid_events:
                best_idx = self._score_candidates(valid_events, sigma_ap, sigma2_ap, displacement_data)
                detected_events.append(best_idx)
                event_intervals.append(interval)
                best_event_group_indices.append(current_group_index)  # 记录该组有最佳事件
                if not silent:
                    print(f"    选中最佳事件 @ {timestamps[best_idx]:.3f}s (组大小={len(event_group)}, 持续时间={duration:.2f}s)")
            else:
                # 没有最佳事件，但区间已记录到all_event_intervals
                if not silent:
                    print(f"    组被丢弃 (无有效事件, 持续时间={duration:.2f}s)")
            
            event_group = []
            last_candidate_time = None
        
        for i in range(len(sigma_ap)):
            current_time = timestamps[i]
            volatility_buffer.append(sigma_ap[i])
            if len(volatility_buffer) > 3:
                volatility_buffer.pop(0)
            
            is_local_max = False
            if len(volatility_buffer) == 3:
                a, b, c = volatility_buffer
                is_local_max = (a <= b) and (b > c)
            
            if is_local_max and len(volatility_buffer) == 3 and volatility_buffer[-2] >= self.H and sigma2_ap[i] <= self.H2:
                candidate_idx = max(0, i - 1)
                candidate_time = timestamps[candidate_idx]
                
                if not event_group and not silent:
                    print(f"  新组开始 @ {candidate_time:.3f}s")
                    
                if not event_group or event_group[-1] != candidate_idx:
                    event_group.append(candidate_idx)
                    last_candidate_time = candidate_time  # 更新最后一个候选的时间（滑动窗口关键）
            
            # 检查是否需要处理当前组（使用滑动窗口：从最后一个候选开始计算）
            if event_group and last_candidate_time is not None:
                should_process = False
                
                if self.suppression_enabled:
                    # 启用时间窗口抑制：距离最后一个候选超过窗口时间才处理
                    # 这是滑动窗口：如果持续有新候选加入，窗口会一直延后
                    if (current_time - last_candidate_time) >= self.suppression_window:
                        should_process = True
                        if not silent:
                            print(f"  处理组 ({len(event_group)} 个候选) - 距最后候选超过抑制窗口")
                else:
                    # 禁用时间窗口抑制：每个候选都作为独立事件立即处理
                    should_process = True
                    if not silent:
                        print(f"  处理组 ({len(event_group)} 个候选) - 抑制窗口已禁用")
                
                if should_process:
                    process_group()
        
        # 处理最后一组
        if event_group:
            if not silent:
                print(f"  处理最后一组 ({len(event_group)} 个候选)")
            process_group(final=True)
        
        if not silent:
            print(f"组内最优检测完成: {len(detected_events)} 个事件, {len(all_event_intervals)} 个事件组")
        return np.array(detected_events), event_intervals, event_group_candidates, all_event_intervals, best_event_group_indices
    
    def process_data(self, timestamps, signal_data, signal_data_y=None, invert_z=True, silent=False):
        """
        处理数据并检测事件（双轴模式）
        
        参数:
            timestamps: 时间戳序列
            signal_data: Z轴信号数据（位移）
            signal_data_y: Y轴信号数据（位移，必须提供）
            invert_z: 是否对Z轴取反（默认True）
            silent: 是否静默模式（不打印日志，用于实时检测）
        
        返回:
            QVDetectionResult: 检测结果
        """
        # 更新采样频率
        if len(timestamps) > 1:
            dt_actual = np.mean(np.diff(timestamps))
            self.fs = 1.0 / dt_actual
            self.dt = dt_actual
            if not silent:
                print(f"采样频率: {self.fs:.2f} Hz")
        
        # 双轴模式检测（Y轴+Z轴）
        if signal_data_y is None:
            raise ValueError("双轴检测需要Y轴数据 (signal_data_y)")
        
        if not silent:
            print("=" * 50)
            print("双轴模式检测")
            print("=" * 50)
        
        # 0. 使用卡尔曼滤波将位移转换为加速度（两个轴分别计算）
        if not silent:
            print("使用卡尔曼滤波计算Y轴加速度...")
        acceleration_y = self._displacement_to_acceleration_kalman(signal_data_y, timestamps)
        if not silent:
            print(f"Y轴加速度统计: 均值={np.mean(acceleration_y):.3f}, 标准差={np.std(acceleration_y):.3f}")
        
        if not silent:
            print("使用卡尔曼滤波计算Z轴加速度...")
        # Z轴根据invert_z参数决定是否取反
        signal_z_processed = -signal_data if invert_z else signal_data
        acceleration_z = self._displacement_to_acceleration_kalman(signal_z_processed, timestamps)
        if not silent:
            print(f"Z轴加速度统计: 均值={np.mean(acceleration_z):.3f}, 标准差={np.std(acceleration_z):.3f}")
            if invert_z:
                print("  (Z轴已取反)")
        
        # 1. 计算两个轴的QV
        if not silent:
            print("计算Y轴二次变分...")
        qv_y, k_star_y = self.calculate_qv(acceleration_y, k0=self.k0)
        if not silent:
            print(f"Y轴最优子采样步长 k* = {k_star_y}")
        
        if not silent:
            print("计算Z轴二次变分...")
        qv_z, k_star_z = self.calculate_qv(acceleration_z, k0=self.k0)
        if not silent:
            print(f"Z轴最优子采样步长 k* = {k_star_z}")
        
        # 2. 估计两个轴的波动率
        if not silent:
            print("估计Y轴波动率...")
        sigma_y, sigma2_y = self.estimate_volatility(qv_y)
        
        if not silent:
            print("估计Z轴波动率...")
        sigma_z, sigma2_z = self.estimate_volatility(qv_z)
        
        # 3. 双轴平均（与QV_RT_OUR.py的detect_swallows方法一致）
        # 规则：如果任一轴数据为0，则加和结果置零
        if not silent:
            print("计算双轴平均波动率...")
        
        # 创建掩码：标记Y轴或Z轴为0的位置（使用可配置阈值）
        y_zero_mask = np.abs(signal_data_y) < self.zero_threshold
        z_zero_mask = np.abs(signal_z_processed) < self.zero_threshold
        either_zero_mask = y_zero_mask | z_zero_mask
        
        zero_count = np.sum(either_zero_mask)
        if zero_count > 0 and not silent:
            print(f"⚠ 检测到 {zero_count}/{len(signal_data_y)} 个数据点中有轴为零")
            print(f"   这些位置的加和将被置零")
        
        # 计算平均波动率，但在任一轴为0处置零
        sigma = (sigma_y + sigma_z) / 2
        sigma[either_zero_mask] = 0.0
        
        sigma2 = (sigma2_y + sigma2_z) / 2
        sigma2[either_zero_mask] = 0.0
        
        # 使用平均后的位移数据进行验证，任一轴为0则置零
        combined_signal = (signal_data_y + signal_z_processed) / 2
        combined_signal[either_zero_mask] = 0.0
        
        # 4. 使用组内最优检测
        if not silent:
            print("使用组内最优检测（双轴平均）...")
        # 传递silent参数给组内检测，接收扩展的返回值
        events, event_intervals, event_candidates, all_event_intervals, best_event_group_indices = self.detect_swallows_groupwise_best(
            sigma, sigma2, timestamps, combined_signal, silent=silent
        )
        
        if not silent:
            print(f"最终检测到 {len(events)} 个事件")
        
        # 扩展结果类（应用零值掩码）
        combined_qv = (qv_y + qv_z) / 2
        combined_qv[either_zero_mask] = 0.0
        
        combined_acceleration = (acceleration_y + acceleration_z) / 2
        combined_acceleration[either_zero_mask] = 0.0
        
        result = QVDetectionResult(
            events=events,
            event_times=[timestamps[i] for i in events],
            volatility=sigma,
            qv=combined_qv,
            signal=combined_signal
        )
        result.event_intervals = event_intervals
        result.event_candidates = event_candidates
        result.all_event_intervals = all_event_intervals
        result.best_event_group_indices = best_event_group_indices
        result.acceleration = combined_acceleration
        # 保存各轴数据用于可视化
        result.signal_y = signal_data_y
        result.signal_z = signal_data
        result.sigma_y = sigma_y
        result.sigma_z = sigma_z
        result.qv_y = qv_y
        result.qv_z = qv_z
        result.acceleration_y = acceleration_y
        result.acceleration_z = acceleration_z
        
        return result
