"""
安全吞咽容量(SSC)计算系统 - 基于HDBSCAN和核密度估计
根据上次喂食量、次数调整吞咽时间窗和计算规则喂食因子

核心模块：提供训练、分析和模型保存加载功能
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Union
import pickle
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
import os


@dataclass
class TrainingConfig:
    """训练配置参数"""
    min_cluster_size: int = 10
    kde_bandwidth: float = 0.5
    T_max: float = 5.0
    epsilon: float = 0.01
    gamma: float = 0.8
    beta: float = 1.0
    population: str = 'normal'
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TrainingConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass 
class TrainingResult:
    """训练结果"""
    success: bool = False
    message: str = ""
    model_path: Optional[str] = None
    cluster_count: int = 0
    sample_count: int = 0
    training_time: float = 0.0
    timestamp: str = ""
    config: Optional[TrainingConfig] = None
    summary: Dict = field(default_factory=dict)


class DataLoader:
    """数据加载器，支持多种CSV格式"""
    
    SUPPORTED_FORMATS = {
        'track_swallow': ['timestamps', 'displacements'],
        'tracking_data': ['timestamps', 'y_displacement', 'z_displacement', 'displacements'],
        'custom': []  # 用户自定义列
    }
    
    @staticmethod
    def load_csv(file_path: str, volume_col: str = None, duration_col: str = None,
                 feeding_id_col: str = None, auto_detect: bool = True) -> Dict:
        """
        加载CSV文件并提取训练数据
        
        参数:
            file_path: CSV文件路径
            volume_col: 体积列名（可选，若不指定则自动推断）
            duration_col: 时长列名（可选）
            feeding_id_col: 喂食ID列名（可选）
            auto_detect: 是否自动检测数据格式
            
        返回:
            包含volumes, durations, feeding_ids的字典
        """
        df = pd.read_csv(file_path)
        
        result = {
            'dataframe': df,
            'volumes': None,
            'durations': None,
            'feeding_ids': None,
            'detected_format': None,
            'columns': list(df.columns)
        }
        
        if auto_detect:
            result['detected_format'] = DataLoader._detect_format(df)
            
        # 如果有明确的列名指定，使用指定的列
        if volume_col and volume_col in df.columns:
            result['volumes'] = df[volume_col].values
        if duration_col and duration_col in df.columns:
            result['durations'] = df[duration_col].values
        if feeding_id_col and feeding_id_col in df.columns:
            result['feeding_ids'] = df[feeding_id_col].values
            
        # 如果没有指定，尝试自动推断
        if result['volumes'] is None:
            result['volumes'] = DataLoader._infer_volume_column(df)
        if result['durations'] is None:
            result['durations'] = DataLoader._infer_duration_column(df)
        if result['feeding_ids'] is None:
            result['feeding_ids'] = DataLoader._generate_feeding_ids(len(df))
            
        return result
    
    @staticmethod
    def _detect_format(df: pd.DataFrame) -> str:
        """检测CSV格式类型"""
        cols = set(df.columns)
        
        for format_name, required_cols in DataLoader.SUPPORTED_FORMATS.items():
            if format_name == 'custom':
                continue
            if set(required_cols).issubset(cols):
                return format_name
        return 'custom'
    
    @staticmethod
    def _infer_volume_column(df: pd.DataFrame) -> np.ndarray:
        """推断体积列"""
        # 优先查找包含volume/displacement的列
        for col in df.columns:
            col_lower = col.lower()
            if 'volume' in col_lower or 'displacement' in col_lower:
                return np.abs(df[col].values)
        # 使用第二列作为默认
        if len(df.columns) > 1:
            return np.abs(df.iloc[:, 1].values)
        return np.ones(len(df))
    
    @staticmethod
    def _infer_duration_column(df: pd.DataFrame) -> np.ndarray:
        """推断时长列（从时间戳差值计算）"""
        for col in df.columns:
            col_lower = col.lower()
            if 'duration' in col_lower:
                return np.abs(df[col].values)
            if 'time' in col_lower or 'stamp' in col_lower:
                timestamps = df[col].values
                # 计算时间间隔
                durations = np.diff(timestamps, prepend=timestamps[0])
                durations[0] = durations[1] if len(durations) > 1 else 1.0
                return np.abs(durations)
        # 默认返回均匀时长
        return np.ones(len(df))
    
    @staticmethod
    def _generate_feeding_ids(n_samples: int, samples_per_feeding: int = 50) -> np.ndarray:
        """生成喂食ID"""
        n_feedings = max(1, n_samples // samples_per_feeding)
        return np.repeat(np.arange(n_feedings), samples_per_feeding)[:n_samples]
    
    @staticmethod
    def preprocess_data(volumes: np.ndarray, durations: np.ndarray, 
                       feeding_ids: np.ndarray,
                       remove_zeros: bool = True,
                       min_volume: float = 0.01,
                       min_duration: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预处理数据
        
        参数:
            volumes: 原始体积数组
            durations: 原始时长数组  
            feeding_ids: 喂食ID数组
            remove_zeros: 是否移除零值
            min_volume: 最小体积阈值
            min_duration: 最小时长阈值
            
        返回:
            处理后的 (volumes, durations, feeding_ids)
        """
        mask = np.ones(len(volumes), dtype=bool)
        
        if remove_zeros:
            mask &= (volumes > min_volume)
            mask &= (durations > min_duration)
        
        # 处理NaN和Inf
        mask &= np.isfinite(volumes)
        mask &= np.isfinite(durations)
        
        return volumes[mask], durations[mask], feeding_ids[mask]


class SwallowAnalyzer:
    """吞咽行为分析器"""
    
    def __init__(self, min_cluster_size=10, kde_bandwidth=0.5, T_max=5.0, epsilon=0.01, 
                 gamma=0.8, beta=1.0):
        """
        参数:
            min_cluster_size: HDBSCAN最小簇大小
            kde_bandwidth: 核密度估计带宽
            T_max: 最大吞咽时间阈值(秒)
            epsilon: 动态时间窗函数中的小常数
            gamma: 患者、老年人参考调整系数
            beta: 惩罚系数
        """
        self.min_cluster_size = min_cluster_size
        self.kde_bandwidth = kde_bandwidth
        self.T_max = T_max
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta
        
        self.scaler = StandardScaler()
        self.clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
        self.cluster_results = {}
        self._trained = False
        self._training_data = None
        self._results = None
    
    @classmethod
    def from_config(cls, config: TrainingConfig) -> 'SwallowAnalyzer':
        """从配置创建分析器"""
        return cls(
            min_cluster_size=config.min_cluster_size,
            kde_bandwidth=config.kde_bandwidth,
            T_max=config.T_max,
            epsilon=config.epsilon,
            gamma=config.gamma,
            beta=config.beta
        )
    
    def get_config(self) -> TrainingConfig:
        """获取当前配置"""
        return TrainingConfig(
            min_cluster_size=self.min_cluster_size,
            kde_bandwidth=self.kde_bandwidth,
            T_max=self.T_max,
            epsilon=self.epsilon,
            gamma=self.gamma,
            beta=self.beta
        )
    
    def save_model(self, save_path: str, include_data: bool = False) -> str:
        """
        保存训练好的模型
        
        参数:
            save_path: 保存路径（不含扩展名）
            include_data: 是否包含训练数据
            
        返回:
            实际保存的文件路径
        """
        if not self._trained:
            raise ValueError("模型尚未训练，请先调用 train() 或 analyze_swallows()")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'config': self.get_config().to_dict(),
            'scaler': self.scaler,
            'clusterer': self.clusterer,
            'cluster_results': self.cluster_results,
            'results': self._results,
            'trained': self._trained,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        if include_data and self._training_data is not None:
            model_data['training_data'] = self._training_data
        
        # 保存为pickle
        pkl_path = str(save_path) + '.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # 同时保存一份可读的JSON配置
        json_path = str(save_path) + '_config.json'
        json_data = {
            'config': model_data['config'],
            'timestamp': model_data['timestamp'],
            'version': model_data['version'],
            'cluster_summary': {}
        }
        
        if self._results and 'cluster_info' in self._results:
            for k, info in self._results['cluster_info'].items():
                json_data['cluster_summary'][str(k)] = {
                    'sample_count': int(info['sample_count']),
                    'SSC_rule': float(info['SSC_rule']),
                    'mean_volume': float(info['mean_volume']),
                    'mean_duration': float(info['mean_duration']),
                    'd_ref': float(info['d_ref'])
                }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return pkl_path
    
    @classmethod
    def load_model(cls, model_path: str) -> 'SwallowAnalyzer':
        """
        加载已保存的模型
        
        参数:
            model_path: 模型文件路径
            
        返回:
            加载的SwallowAnalyzer实例
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        config = TrainingConfig.from_dict(model_data['config'])
        analyzer = cls.from_config(config)
        
        analyzer.scaler = model_data['scaler']
        analyzer.clusterer = model_data['clusterer']
        analyzer.cluster_results = model_data['cluster_results']
        analyzer._results = model_data['results']
        analyzer._trained = model_data['trained']
        
        if 'training_data' in model_data:
            analyzer._training_data = model_data['training_data']
        
        return analyzer
    
    def train(self, volumes: np.ndarray, durations: np.ndarray, 
              feeding_ids: np.ndarray, population: str = 'normal') -> TrainingResult:
        """
        训练模型（analyze_swallows的包装方法）
        
        参数:
            volumes: 吞咽体积数组 (mL)
            durations: 吞咽时长数组 (s)
            feeding_ids: 每次吞咽所属的喂食ID
            population: 人群类型
            
        返回:
            TrainingResult: 训练结果
        """
        start_time = datetime.now()
        
        try:
            # 保存训练数据
            self._training_data = {
                'volumes': volumes.copy(),
                'durations': durations.copy(),
                'feeding_ids': feeding_ids.copy(),
                'population': population
            }
            
            # 执行分析
            results = self.analyze_swallows(volumes, durations, feeding_ids, population)
            self._results = results
            self._trained = True
            
            # 构建训练结果
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            cluster_count = len(results.get('cluster_info', {}))
            
            summary = {
                'cluster_count': cluster_count,
                'noise_count': int(np.sum(results['cluster_labels'] == -1)),
                'clusters': {}
            }
            
            for k, info in results.get('cluster_info', {}).items():
                summary['clusters'][str(k)] = {
                    'sample_count': int(info['sample_count']),
                    'SSC_rule': float(info['SSC_rule']),
                    'mean_volume': float(info['mean_volume']),
                    'mean_duration': float(info['mean_duration'])
                }
            
            return TrainingResult(
                success=True,
                message=f"训练成功！发现 {cluster_count} 个聚类簇",
                cluster_count=cluster_count,
                sample_count=len(volumes),
                training_time=training_time,
                timestamp=end_time.isoformat(),
                config=self.get_config(),
                summary=summary
            )
            
        except Exception as e:
            return TrainingResult(
                success=False,
                message=f"训练失败: {str(e)}",
                timestamp=datetime.now().isoformat(),
                config=self.get_config()
            )
    
    def train_from_csv(self, csv_path: str, volume_col: str = None, 
                       duration_col: str = None, feeding_id_col: str = None,
                       population: str = 'normal',
                       preprocess: bool = True) -> TrainingResult:
        """
        从CSV文件训练模型
        
        参数:
            csv_path: CSV文件路径
            volume_col: 体积列名
            duration_col: 时长列名
            feeding_id_col: 喂食ID列名
            population: 人群类型
            preprocess: 是否预处理数据
            
        返回:
            TrainingResult: 训练结果
        """
        try:
            # 加载数据
            data = DataLoader.load_csv(
                csv_path, 
                volume_col=volume_col,
                duration_col=duration_col,
                feeding_id_col=feeding_id_col
            )
            
            volumes = data['volumes']
            durations = data['durations']
            feeding_ids = data['feeding_ids']
            
            if volumes is None or durations is None:
                return TrainingResult(
                    success=False,
                    message="无法从CSV文件中提取有效数据",
                    timestamp=datetime.now().isoformat()
                )
            
            # 预处理
            if preprocess:
                volumes, durations, feeding_ids = DataLoader.preprocess_data(
                    volumes, durations, feeding_ids
                )
            
            if len(volumes) < self.min_cluster_size:
                return TrainingResult(
                    success=False,
                    message=f"有效数据点数量({len(volumes)})少于最小簇大小({self.min_cluster_size})",
                    timestamp=datetime.now().isoformat()
                )
            
            # 训练
            result = self.train(volumes, durations, feeding_ids, population)
            result.message = f"从 {csv_path} 加载数据并" + result.message
            
            return result
            
        except Exception as e:
            return TrainingResult(
                success=False,
                message=f"从CSV训练失败: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    def get_results(self) -> Optional[Dict]:
        """获取训练结果"""
        return self._results
    
    def is_trained(self) -> bool:
        """检查模型是否已训练"""
        return self._trained
        
    def prepare_features(self, volumes: np.ndarray, durations: np.ndarray, 
                        count_per_feeding: np.ndarray) -> np.ndarray:
        """
        准备特征向量
        
        参数:
            volumes: 每次吞咽的体积 (mL)
            durations: 每次吞咽的时长 (s)
            count_per_feeding: 每次喂食对应的吞咽次数
            
        返回:
            X: 特征矩阵 [log(v), log(t)]
        """
        # 按照算法: x_{e,j} = [log v_{e,j}, log t_{e,j}]
        log_volumes = np.log(volumes + 1e-10)  # 避免log(0)
        log_durations = np.log(durations + 1e-10)
        
        X = np.column_stack([log_volumes, log_durations])
        return X
    
    def fit_hdbscan(self, X: np.ndarray) -> np.ndarray:
        """
        执行HDBSCAN聚类
        
        参数:
            X: 特征矩阵
            
        返回:
            cluster_labels: 聚类标签 (噪声点为-1)
        """
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # HDBSCAN聚类
        cluster_labels = self.clusterer.fit_predict(X_scaled)
        
        return cluster_labels
    
    def compute_kde_density(self, cluster_data: np.ndarray) -> Tuple[np.ndarray, KernelDensity]:
        """
        计算簇内核密度估计
        
        参数:
            cluster_data: 簇内数据点 [v, t]
            
        返回:
            max_density_point: 最大密度点
            kde: 拟合的KDE模型
        """
        # 高斯核密度估计
        # 带宽矩阵 H_k = n_k^{-1/(d+4)} I_d, d=2
        n_k = len(cluster_data)
        d = 2
        bandwidth = n_k ** (-1 / (d + 4))
        
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(cluster_data)
        
        # 在数据范围内网格搜索最大密度点
        v_min, v_max = cluster_data[:, 0].min(), cluster_data[:, 0].max()
        t_min, t_max = cluster_data[:, 1].min(), cluster_data[:, 1].max()
        
        grid_v = np.linspace(v_min, v_max, 100)
        grid_t = np.linspace(t_min, t_max, 100)
        V, T = np.meshgrid(grid_v, grid_t)
        grid_points = np.vstack([V.ravel(), T.ravel()]).T
        
        # 计算密度
        log_dens = kde.score_samples(grid_points)
        max_idx = np.argmax(log_dens)
        max_density_point = grid_points[max_idx]
        
        return max_density_point, kde
    
    def compute_dynamic_time_window(self, durations: np.ndarray) -> np.ndarray:
        """
        计算动态时间窗调整因子
        
        参数:
            durations: 吞咽时长数组
            
        返回:
            f_t: 动态时间窗因子数组
        """
        # f_{t,i} = T_{max}/T_i if T_i > T_max, else 1
        f_t = np.where(durations > self.T_max, self.T_max / durations, 1.0)
        return f_t
    
    def compute_ssc_rule(self, V_perc: float, C_i: int, f_t: np.ndarray, v_values: np.ndarray) -> float:
        """
        计算规则喂食标准化吞咽容量
        根据新算法: SSC_e = (1/N_e) * sum_{j=1}^{N_e}(v_{e,j} * f_{t,e,j})
        
        参数:
            V_perc: 参考体积(百分位数或最大密度点，本方法中未直接使用)
            C_i: 本次喂食的吞咽次数
            f_t: 动态时间窗因子数组
            v_values: 吞咽体积数组
            
        返回:
            SSC_rule: 规则喂食SSC (单位: mL)
        """
        # 算法(4): SSC_e = (1/N_e) * sum(v_{e,j} * f_{t,e,j})
        N_e = C_i
        SSC = (1 / N_e) * np.sum(v_values * f_t)
        return SSC
    
    def compute_feeding_factors(self, d_new: float, d_ref: float) -> float:
        """
        计算喂食调整因子（根据图片算法(5)）
        
        新算法统一使用:
        - alpha_new = min(1, d_new / d_ref)
        
        参数:
            d_new: 新输入的喂食量密度估计 d_new = KDE_e(x)
            d_ref: 参考密度（峰值密度 d_ref^k）
            
        返回:
            alpha_new: 调整因子，取值范围 (0, 1]
        """
        # 算法(5): 喂食调整因子
        # d_new = KDE_e(x) 为新样本的密度估计
        # alpha_new = min(1, d_new / d_ref)
        if d_ref > 0:
            alpha_new = min(1.0, d_new / d_ref)
        else:
            alpha_new = 1.0
            
        return alpha_new
    
    def estimate_density_for_sample(self, vol: float, dur: float, cluster_info: Dict) -> Tuple[float, int, Dict[int, float]]:
        """
        估算新样本在每个簇下的 KDE 密度并返回最大密度值（d_new）
        
        参数:
            vol: 新样本体积（mL，与 compute_kde_density 使用的原始数据域一致）
            dur: 新样本时长（s）
            cluster_info: analyze_swallows 返回的簇信息字典（包含每簇的 'kde'）
            
        返回:
            d_new: 选取的最大密度值（若无可用 KDE 则返回 0.0）
            best_k: 对应的簇 id（找不到则返回 None）
            densities: 每个簇的密度字典 {cluster_id: density_value}
        """
        x = np.array([[vol, dur]])
        densities = {}
        
        for k, info in cluster_info.items():
            kde = info.get('kde')
            if kde is None:
                continue
            # KernelDensity.score_samples 返回 log 密度，需要 exp 得到密度值
            log_d = kde.score_samples(x)[0]
            densities[k] = float(np.exp(log_d))
        
        if not densities:
            return 0.0, None, {}
        
        best_k = max(densities, key=densities.get)
        d_new = densities[best_k]
        return d_new, best_k, densities
    
    def compute_adjusted_ssc(self, base_ssc: float, d_new: float, d_ref: float, 
                           population: str = 'normal') -> Dict:
        """
        使用喂食调整因子调整基础 SSC（根据新算法）
        
        算法流程:
        1. 计算 alpha_new = min(1, d_new / d_ref)
        2. 根据人群类型确定 gamma_final:
           - normal: gamma_final = alpha_new
           - elderly: gamma_final = gamma_base * alpha_new  (gamma_base=0.8)
           - patient: gamma_final = gamma_base * gamma_penalty (beta系数)
        3. SSC_new = SSC_e * gamma_final
        
        参数:
            base_ssc: 基础SSC（簇级或个体化）SSC_e
            d_new: 新样本的密度估计 KDE_e(x)
            d_ref: 参考密度（峰值密度 d_ref^k）
            population: 人群类型 ('normal', 'elderly', 'patient')
            
        返回:
            包含调整后SSC及因子的字典
        """
        # 计算基础调整因子
        alpha_new = self.compute_feeding_factors(d_new, d_ref)
        
        # 根据人群类型应用不同的调整策略（参考图片中的逻辑）
        if population.lower() == 'normal':
            # 正常人：gamma_final = alpha_new
            gamma_final = alpha_new
            gamma_label = 'gamma_detect'
        elif population.lower() == 'elderly':
            # 老年人：gamma_final = gamma_base * alpha_new
            gamma_final = self.gamma * alpha_new
            gamma_label = 'gamma_low * gamma_penalty'
        else:  # patient
            # 患者：gamma_final = gamma_base * gamma_penalty
            # 这里 gamma_penalty 可以基于 alpha_new 和 beta 系数调整
            # 根据图片算法，使用更保守的策略
            delta = max(0, (d_ref - d_new) / d_ref)  # 密度偏差
            gamma_penalty = min(1.0, d_new / d_ref) if d_ref > 0 else 1.0
            gamma_final = self.gamma * gamma_penalty * (1 - self.beta * delta)
            gamma_final = max(0.1, gamma_final)  # 确保不低于最小阈值
            gamma_label = 'gamma_base * gamma_penalty'
        
        # 最终SSC: SSC_new = SSC_e * gamma_final
        ssc_adjusted = base_ssc * gamma_final
        
        return {
            'base_ssc': base_ssc,
            'alpha_new': alpha_new,
            'gamma_final': gamma_final,
            'gamma_label': gamma_label,
            'ssc_adjusted': ssc_adjusted,
            'd_new': d_new,
            'd_ref': d_ref,
            'population': population,
            'delta': max(0, (d_ref - d_new) / d_ref) if d_ref > 0 else 0
        }
    
    def compute_personalized_ssc(self, vol: float, dur: float, cluster_id: int, 
                                 results: Dict, percentile: float = 75.0, 
                                 apply_population_factor: bool = True) -> Dict:
        """
        根据新样本的特征计算个体化的 SSC 推荐（解决簇内"一刀切"问题）
        
        参数:
            vol: 新样本体积（mL）
            dur: 新样本时长（s）
            cluster_id: 所属簇 ID
            results: analyze_swallows 返回的结果字典
            percentile: 安全百分位数（默认 75，即推荐不超过该簇内 75% 样本的体积）
            apply_population_factor: 是否应用人群保守系数
            
        返回:
            包含个体化 SSC 及详细信息的字典
        """
        cluster_labels = results['cluster_labels']
        volumes = results['volumes']
        durations = results['durations']
        cluster_info = results['cluster_info']
        
        if cluster_id not in cluster_info:
            return {'error': f'Cluster {cluster_id} not found'}
        
        # 获取该簇的所有样本
        mask = cluster_labels == cluster_id
        cluster_vols = volumes[mask]
        cluster_durs = durations[mask]
        
        # 计算该样本的密度相对位置（与峰值密度的比值）
        kde = cluster_info[cluster_id]['kde']
        x = np.array([[vol, dur]])
        log_d_sample = kde.score_samples(x)[0]
        d_sample = np.exp(log_d_sample)
        
        peak_point = cluster_info[cluster_id]['max_density_point']
        log_d_peak = kde.score_samples(np.array([peak_point]))[0]
        d_peak = np.exp(log_d_peak)
        
        density_ratio = d_sample / d_peak if d_peak > 0 else 0.0
        
        # 方法1：基于体积分位数（考虑时长权重）
        # 找到与新样本时长相近的样本子集
        dur_tolerance = 0.5  # 时长容差（秒）
        similar_dur_mask = np.abs(cluster_durs - dur) <= dur_tolerance
        
        if similar_dur_mask.sum() >= 5:  # 如果有足够多相似时长样本
            similar_vols = cluster_vols[similar_dur_mask]
            ssc_percentile = np.percentile(similar_vols, percentile)
        else:  # 否则用全簇
            ssc_percentile = np.percentile(cluster_vols, percentile)
        
        # 获取簇的基准安全值（SSC_rule 或 均值）
        cluster_ssc_rule = cluster_info[cluster_id]['SSC_rule']
        cluster_mean_vol = cluster_info[cluster_id]['mean_volume']
        
        # 方法2：基于密度调整的保守估计
        # 密度越低（越边缘），越保守（降低推荐量）
        # 重要修正：使用簇的基准值而非用户输入值，避免迭代衰减
        safety_factor = 0.7 + 0.3 * density_ratio  # 范围 [0.7, 1.0]
        ssc_density_adjusted = cluster_ssc_rule * safety_factor  # 使用簇的SSC_rule作为基准
        
        # 方法3：动态时间窗调整
        # 使用簇的均值作为基准，根据时长进行调整
        f_t = self.T_max / dur if dur > self.T_max else 1.0
        ssc_time_adjusted = cluster_mean_vol * f_t  # 使用簇均值作为基准
        
        # 综合推荐：取较保守值
        ssc_recommended = min(ssc_percentile, ssc_density_adjusted, ssc_time_adjusted)
        
        # 根据人群类型应用保守系数（如果启用）
        population = cluster_info[cluster_id].get('population', 'normal')
        if apply_population_factor and population.lower() in ('elderly', 'patient'):
            ssc_recommended_adjusted = ssc_recommended * self.gamma
        else:
            ssc_recommended_adjusted = ssc_recommended
        
        return {
            'cluster_id': cluster_id,
            'input_volume': vol,
            'input_duration': dur,
            'density_ratio': density_ratio,
            'ssc_percentile_based': ssc_percentile,
            'ssc_density_adjusted': ssc_density_adjusted,
            'ssc_time_adjusted': ssc_time_adjusted,
            'ssc_recommended': ssc_recommended,
            'ssc_recommended_adjusted': ssc_recommended_adjusted,
            'safety_factor': safety_factor,
            'population': population,
            'cluster_ssc_rule': cluster_ssc_rule,  # 簇的SSC规则值（基准）
            'cluster_vol_mean': cluster_info[cluster_id]['mean_volume'],
            'cluster_vol_std': np.std(cluster_vols),
            'cluster_dur_mean': cluster_info[cluster_id]['mean_duration'],
            'cluster_dur_std': np.std(cluster_durs),
        }
    
    def analyze_swallows(self, volumes: np.ndarray, durations: np.ndarray, 
                        feeding_ids: np.ndarray, population: str = 'normal') -> Dict:
        """
        完整的吞咽分析流程
        
        参数:
            volumes: 吞咽体积数组 (mL)
            durations: 吞咽时长数组 (s)
            feeding_ids: 每次吞咽所属的喂食ID
            population: 人群类型 ('normal', 'elderly', 'patient')，影响保守系数应用
            
        返回:
            results: 包含聚类、SSC、调整因子等结果的字典
        """
        # 1. 准备特征
        X = self.prepare_features(volumes, durations, feeding_ids)
        
        # 2. HDBSCAN聚类
        cluster_labels = self.fit_hdbscan(X)
        
        # 3. 对每个簇计算KDE和最大密度点
        unique_clusters = np.unique(cluster_labels)
        cluster_info = {}
        
        for k in unique_clusters:
            if k == -1:  # 跳过噪声点
                continue
                
            # 获取该簇的原始数据
            mask = cluster_labels == k
            cluster_volumes = volumes[mask]
            cluster_durations = durations[mask]
            cluster_data = np.column_stack([cluster_volumes, cluster_durations])
            
            # 计算最大密度点
            max_point, kde = self.compute_kde_density(cluster_data)
            
            # 计算动态时间窗因子
            f_t = self.compute_dynamic_time_window(cluster_durations)
            
            # 计算该簇的规则SSC
            V_perc = max_point[0]  # 使用最大密度点的体积
            C_i = len(cluster_volumes)
            SSC_rule = self.compute_ssc_rule(V_perc, C_i, f_t, cluster_volumes)
            
            # 计算该簇峰值密度（用于后续调整因子计算）
            d_ref = np.exp(kde.score_samples(np.array([max_point]))[0])
            
            cluster_info[k] = {
                'max_density_point': max_point,
                'sample_count': len(cluster_data),
                'kde': kde,
                'SSC_rule': SSC_rule,
                'd_ref': d_ref,  # 保存参考密度
                'mean_volume': np.mean(cluster_volumes),
                'mean_duration': np.mean(cluster_durations),
                'population': population
            }
        
        results = {
            'cluster_labels': cluster_labels,
            'cluster_info': cluster_info,
            'X': X,
            'volumes': volumes,
            'durations': durations
        }
        
        return results
    
    def visualize_results(self, results: Dict):
        """可视化分析结果"""
        cluster_labels = results['cluster_labels']
        volumes = results['volumes']
        durations = results['durations']
        cluster_info = results['cluster_info']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 图1: 聚类结果
        scatter = ax1.scatter(volumes, durations, c=cluster_labels,
                              cmap='tab20', s=30, alpha=0.6)

        # 为每个簇绘制KDE等高线
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for idx, (k, info) in enumerate(cluster_info.items()):
            kde = info.get('kde')
            if kde is None:
                continue
            
            # 获取该簇的数据范围
            mask = cluster_labels == k
            cluster_vols = volumes[mask]
            cluster_durs = durations[mask]
            
            # 创建网格用于绘制等高线
            v_min, v_max = cluster_vols.min(), cluster_vols.max()
            t_min, t_max = cluster_durs.min(), cluster_durs.max()
            
            # 扩展边界以便更好地显示
            v_range = v_max - v_min
            t_range = t_max - t_min
            v_min -= v_range * 0.1
            v_max += v_range * 0.1
            t_min -= t_range * 0.1
            t_max += t_range * 0.1
            
            v_grid = np.linspace(v_min, v_max, 100)
            t_grid = np.linspace(t_min, t_max, 100)
            V_mesh, T_mesh = np.meshgrid(v_grid, t_grid)
            grid_points = np.column_stack([V_mesh.ravel(), T_mesh.ravel()])
            
            # 计算网格上的密度
            log_dens = kde.score_samples(grid_points)
            dens = np.exp(log_dens).reshape(V_mesh.shape)
            
            # 绘制等高线（只绘制几条关键等高线）
            color = colors[idx % len(colors)]
            contour = ax1.contour(V_mesh, T_mesh, dens, levels=5, 
                                 colors=color, alpha=0.4, linewidths=1.5)
            ax1.clabel(contour, inline=True, fontsize=8, fmt='%.3f')

        # 标出每个簇的最大密度点
        for k, info in cluster_info.items():
            v_max, t_max = info['max_density_point']
            ax1.plot(v_max, t_max, 'r*', markersize=15, mew=2)
            ax1.text(v_max, t_max, f'  Cluster {k}', fontsize=10, fontweight='bold')

        ax1.set_xlabel('Swallow Volume (mL)', fontsize=12)
        ax1.set_ylabel('Swallow Duration (s)', fontsize=12)
        ax1.set_title('HDBSCAN Clustering and KDE Peaks with Density Contours', fontsize=14)
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Cluster ID (-1 = Noise)')

        # 图2: SSC 分布（改为箱线图显示簇内变异）
        clusters = list(cluster_info.keys())
        sample_counts = [cluster_info[k]['sample_count'] for k in clusters]
        
        # 计算每个样本的"单次 SSC" = v_i * f_t_i
        f_t_all = self.compute_dynamic_time_window(durations)
        per_sample_ssc = volumes * f_t_all
        
        # 为每个簇收集其样本的 SSC 值
        per_cluster_ssc_samples = []
        for k in clusters:
            mask = cluster_labels == k
            cluster_ssc = per_sample_ssc[mask]
            per_cluster_ssc_samples.append(cluster_ssc if len(cluster_ssc) > 0 else np.array([np.nan]))
        
        # 绘制箱线图
        bp = ax2.boxplot(per_cluster_ssc_samples, positions=range(len(clusters)),
                         widths=0.6, patch_artist=True, showfliers=True,
                         flierprops=dict(marker='o', markersize=3, alpha=0.4))
        
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.6)

        ax2.set_xlabel('Cluster ID', fontsize=12)
        ax2.set_ylabel('Per-sample SSC (mL)', fontsize=12)
        ax2.set_title('SSC Distribution per Cluster (boxplot) with cluster-rule SSC (◆)', fontsize=13)
        ax2.set_xticks(range(len(clusters)))
        ax2.set_xticklabels([f'Cluster {k}' for k in clusters])
        ax2.grid(True, alpha=0.3, axis='y')

        # 在箱线图上方标注样本数量，并用黑色菱形标记簇规则 SSC
        for i, (k, count) in enumerate(zip(clusters, sample_counts)):
            # 标注样本数
            y_max = np.nanmax(per_cluster_ssc_samples[i])
            if np.isfinite(y_max):
                ax2.text(i, y_max * 1.03,
                         f'n={count}', ha='center', va='bottom', fontsize=9)
            
            # 用黑色菱形标记该簇的规则 SSC（簇平均）
            ssc_rule = cluster_info[k]['SSC_rule']
            ax2.plot(i, ssc_rule, 'kD', markersize=7, label='Cluster-rule SSC' if i == 0 else '_nolegend_')
        
        ax2.legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        plt.show()
    
    def print_summary(self, results: Dict):
        """打印分析摘要"""
        cluster_info = results['cluster_info']
        
        print("\n" + "="*70)
        print("           SSC Analysis Results")
        print("="*70)
        
        for k in sorted(cluster_info.keys()):
            info = cluster_info[k]
            v_max, t_max = info['max_density_point']

            print(f"\n[Cluster {k}]")
            print(f"  Sample Count: {info['sample_count']}")
            print(f"  Peak Density Point: Volume={v_max:.2f} mL, Duration={t_max:.2f} s")
            print(f"  Peak Density (d_ref): {info['d_ref']:.6f}")
            print(f"  SSC_rule (cluster average): {info['SSC_rule']:.2f} mL")
            print(f"  Mean Volume: {info['mean_volume']:.2f} mL")
            print(f"  Mean Duration: {info['mean_duration']:.2f} s")
        
        print("\n" + "="*70)


# -----------------------------
# 主程序：生成模拟数据并分析
# -----------------------------
def main():
    """主函数"""
    np.random.seed(42)
    
    # 1. 生成模拟吞咽数据
    # 模拟多个吞咽模式
    mode1 = np.random.multivariate_normal([2.0, 1.0], [[0.3, 0.1], [0.1, 0.2]], 200)   # 小体积+短时间
    mode2 = np.random.multivariate_normal([4.5, 2.5], [[0.4, 0.05], [0.05, 0.3]], 250)  # 大体积+长时间
    mode3 = np.random.multivariate_normal([3.0, 4.0], [[0.2, 0.0], [0.0, 0.4]], 150)   # 中体积+长时间
    noise = np.random.uniform([0.5, 0.5], [6.0, 5.0], (50, 2))                         # 噪声点
    
    data = np.vstack([mode1, mode2, mode3, noise])
    
    # 确保数据为正值
    volumes = np.abs(data[:, 0])
    durations = np.abs(data[:, 1])
    
    # 模拟喂食ID（假设分为多次喂食）
    feeding_ids = np.repeat(np.arange(10), len(volumes) // 10)
    
    # 2. 创建分析器并分析
    analyzer = SwallowAnalyzer(
        min_cluster_size=10,
        kde_bandwidth=0.5,
        T_max=5.0,
        epsilon=0.01,
        gamma=0.8,
        beta=1.0
    )
    
    results = analyzer.analyze_swallows(volumes, durations, feeding_ids)
    
    # 3. 输出结果
    analyzer.print_summary(results)
    
    # 4. 可视化
    analyzer.visualize_results(results)
    
    # 5. 演示喂食调整因子计算
    if len(results['cluster_info']) > 0:
        print("\n" + "="*70)
        print("           Feeding Adjustment Factors Example")
        print("="*70)
        
        # 用一个示例新喂食样本（vol, dur）通过已拟合的各簇 KDE 估算 d_new
        # 注意：这里传入的 vol, dur 必须与 compute_kde_density 时使用的数据域一致（当前为原始值）
        new_vol, new_dur = 3.2, 2.0
        d_new, best_k, all_dens = analyzer.estimate_density_for_sample(
            new_vol, new_dur, results['cluster_info']
        )
        
        # 使用最佳匹配簇的参考密度
        if best_k is not None:
            d_ref = results['cluster_info'][best_k]['d_ref']
        else:
            d_ref = 0.20  # 默认参考值
        
        alpha_new = analyzer.compute_feeding_factors(d_new, d_ref)
        
        print(f"\nNew feeding sample: Volume={new_vol:.2f} mL, Duration={new_dur:.2f} s")
        print(f"Estimated densities per cluster: {', '.join([f'Cluster {k}: {d:.6f}' for k, d in all_dens.items()])}")
        print(f"Best matching cluster: {best_k}")
        print(f"New feeding density estimate (d_new): {d_new:.6f}")
        print(f"Reference density (d_ref): {d_ref:.6f}")
        print(f"Adjustment factor α_new = min(1, d_new/d_ref): {alpha_new:.4f}")
        
        # 6. 演示个体化 SSC 推荐（解决簇内差异问题）
        print("\n" + "="*70)
        print("           Personalized SSC Recommendations")
        print("="*70)
        
        # 测试不同场景的个体化 SSC
        test_cases = [
            (2.0, 1.0, "小体积+短时长（簇0核心区）"),
            (3.2, 2.0, "中等体积+中等时长（簇间过渡）"),
            (5.0, 3.5, "大体积+长时长（簇1高端）"),
            (3.5, 5.2, "中等体积+超长时长（边缘/异常）"),
        ]
        
        for vol, dur, description in test_cases:
            # 先确定所属簇
            d_new, best_cluster, _ = analyzer.estimate_density_for_sample(
                vol, dur, results['cluster_info']
            )
            
            if best_cluster is not None:
                # 获取该簇的参考密度
                d_ref = results['cluster_info'][best_cluster]['d_ref']
                
                # 计算个体化 SSC
                personalized = analyzer.compute_personalized_ssc(
                    vol, dur, best_cluster, results, percentile=75
                )
                
                print(f"\n【{description}】")
                print(f"  输入: Volume={vol:.1f} mL, Duration={dur:.1f} s")
                print(f"  匹配簇: Cluster {best_cluster}")
                print(f"  密度估计: d_new={d_new:.6f}, d_ref={d_ref:.6f}")
                print(f"  密度比（相对峰值）: {personalized['density_ratio']:.2%}")
                print(f"  簇平均 SSC: {results['cluster_info'][best_cluster]['SSC_rule']:.2f} mL")
                print(f"  个体化基础 SSC: {personalized['ssc_recommended']:.2f} mL")
                print(f"    - 基于分位数: {personalized['ssc_percentile_based']:.2f} mL")
                print(f"    - 密度调整: {personalized['ssc_density_adjusted']:.2f} mL (安全系数={personalized['safety_factor']:.2f})")
                print(f"    - 时间调整: {personalized['ssc_time_adjusted']:.2f} mL")
                
                # 应用喂食调整因子得到最终推荐（分别演示三种人群）
                for pop_type in ['normal', 'elderly', 'patient']:
                    adjusted = analyzer.compute_adjusted_ssc(
                        personalized['ssc_recommended'], d_new, d_ref, population=pop_type
                    )
                    print(f"  ✓ 最终推荐 SSC ({pop_type}): {adjusted['ssc_adjusted']:.2f} mL (γ_final={adjusted['gamma_final']:.3f})")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    main()



