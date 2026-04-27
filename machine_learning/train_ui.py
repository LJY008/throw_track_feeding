"""
SSC训练系统 - UI界面
提供可视化界面用于调整训练参数、导入数据和训练模型
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from pathlib import Path
import sys
import os

# 确保可以导入同目录的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import SwallowAnalyzer, TrainingConfig, DataLoader, TrainingResult


class TrainUI:
    """SSC训练系统UI界面"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SSC训练系统 - 安全吞咽容量分析")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # 初始化变量
        self.csv_path = tk.StringVar()
        self.model_save_path = tk.StringVar()
        self.volume_col = tk.StringVar()
        self.duration_col = tk.StringVar()
        self.feeding_id_col = tk.StringVar()
        
        # 训练参数变量
        self.min_cluster_size = tk.IntVar(value=10)
        self.kde_bandwidth = tk.DoubleVar(value=0.5)
        self.T_max = tk.DoubleVar(value=5.0)
        self.epsilon = tk.DoubleVar(value=0.01)
        self.gamma = tk.DoubleVar(value=0.8)
        self.beta = tk.DoubleVar(value=1.0)
        self.population = tk.StringVar(value='normal')
        self.preprocess_data = tk.BooleanVar(value=True)
        self.include_data_in_model = tk.BooleanVar(value=False)
        
        # 分析器实例
        self.analyzer = None
        self.training_result = None
        self.loaded_data = None
        
        # 创建UI
        self._create_ui()
        
    def _create_ui(self):
        """创建UI布局"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建notebook用于分页
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 创建各个页面
        self._create_data_page()
        self._create_params_page()
        self._create_train_page()
        self._create_model_page()
        
    def _create_data_page(self):
        """创建数据导入页面"""
        data_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(data_frame, text="📁 数据导入")
        
        # CSV文件选择
        file_group = ttk.LabelFrame(data_frame, text="CSV文件选择", padding="10")
        file_group.pack(fill=tk.X, pady=5)
        
        file_row = ttk.Frame(file_group)
        file_row.pack(fill=tk.X)
        
        ttk.Label(file_row, text="文件路径:").pack(side=tk.LEFT)
        ttk.Entry(file_row, textvariable=self.csv_path, width=60).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_row, text="浏览...", command=self._browse_csv).pack(side=tk.LEFT)
        ttk.Button(file_row, text="加载预览", command=self._load_preview).pack(side=tk.LEFT, padx=5)
        
        # 列映射设置
        mapping_group = ttk.LabelFrame(data_frame, text="列映射设置（可选，留空则自动检测）", padding="10")
        mapping_group.pack(fill=tk.X, pady=5)
        
        mapping_grid = ttk.Frame(mapping_group)
        mapping_grid.pack(fill=tk.X)
        
        ttk.Label(mapping_grid, text="体积列:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.volume_combo = ttk.Combobox(mapping_grid, textvariable=self.volume_col, width=25)
        self.volume_combo.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(mapping_grid, text="时长列:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.duration_combo = ttk.Combobox(mapping_grid, textvariable=self.duration_col, width=25)
        self.duration_combo.grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(mapping_grid, text="喂食ID列:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.feeding_combo = ttk.Combobox(mapping_grid, textvariable=self.feeding_id_col, width=25)
        self.feeding_combo.grid(row=1, column=1, padx=5, pady=2)
        
        # 数据预览
        preview_group = ttk.LabelFrame(data_frame, text="数据预览", padding="10")
        preview_group.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建Treeview用于显示数据
        columns = ("index", "volume", "duration", "feeding_id")
        self.data_tree = ttk.Treeview(preview_group, columns=columns, show="headings", height=10)
        
        self.data_tree.heading("index", text="序号")
        self.data_tree.heading("volume", text="体积")
        self.data_tree.heading("duration", text="时长")
        self.data_tree.heading("feeding_id", text="喂食ID")
        
        self.data_tree.column("index", width=60)
        self.data_tree.column("volume", width=120)
        self.data_tree.column("duration", width=120)
        self.data_tree.column("feeding_id", width=100)
        
        scrollbar = ttk.Scrollbar(preview_group, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)
        
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 数据统计
        stats_frame = ttk.Frame(data_frame)
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.data_stats_label = ttk.Label(stats_frame, text="尚未加载数据")
        self.data_stats_label.pack(side=tk.LEFT)
        
    def _create_params_page(self):
        """创建参数设置页面"""
        params_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(params_frame, text="⚙️ 训练参数")
        
        # HDBSCAN参数
        hdbscan_group = ttk.LabelFrame(params_frame, text="HDBSCAN聚类参数", padding="10")
        hdbscan_group.pack(fill=tk.X, pady=5)
        
        self._create_param_row(hdbscan_group, "最小簇大小:", self.min_cluster_size, 
                              "HDBSCAN最小簇大小，较大值产生更少但更稳定的簇", 0, 
                              from_=2, to=100, is_int=True)
        
        self._create_param_row(hdbscan_group, "KDE带宽:", self.kde_bandwidth,
                              "核密度估计带宽，影响密度估计的平滑程度", 1,
                              from_=0.1, to=2.0, resolution=0.1)
        
        # 时间窗参数
        time_group = ttk.LabelFrame(params_frame, text="时间窗参数", padding="10")
        time_group.pack(fill=tk.X, pady=5)
        
        self._create_param_row(time_group, "最大吞咽时间 T_max:", self.T_max,
                              "超过此时间的吞咽将被调整，单位：秒", 0,
                              from_=1.0, to=20.0, resolution=0.5)
        
        self._create_param_row(time_group, "Epsilon:", self.epsilon,
                              "动态时间窗函数中的小常数", 1,
                              from_=0.001, to=0.1, resolution=0.001)
        
        # 人群调整参数
        pop_group = ttk.LabelFrame(params_frame, text="人群调整参数", padding="10")
        pop_group.pack(fill=tk.X, pady=5)
        
        self._create_param_row(pop_group, "Gamma (保守系数):", self.gamma,
                              "老年人/患者的安全调整系数，0-1之间", 0,
                              from_=0.1, to=1.0, resolution=0.05)
        
        self._create_param_row(pop_group, "Beta (惩罚系数):", self.beta,
                              "患者密度偏差惩罚系数", 1,
                              from_=0.1, to=2.0, resolution=0.1)
        
        # 人群类型选择
        pop_type_frame = ttk.Frame(pop_group)
        pop_type_frame.grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=5)
        
        ttk.Label(pop_type_frame, text="人群类型:").pack(side=tk.LEFT, padx=5)
        for pop_val, pop_text in [('normal', '正常人'), ('elderly', '老年人'), ('patient', '患者')]:
            ttk.Radiobutton(pop_type_frame, text=pop_text, variable=self.population, 
                           value=pop_val).pack(side=tk.LEFT, padx=10)
        
        # 其他设置
        other_group = ttk.LabelFrame(params_frame, text="其他设置", padding="10")
        other_group.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(other_group, text="预处理数据（移除零值和无效值）", 
                       variable=self.preprocess_data).pack(anchor=tk.W)
        ttk.Checkbutton(other_group, text="在模型中包含训练数据", 
                       variable=self.include_data_in_model).pack(anchor=tk.W)
        
        # 参数预设
        preset_frame = ttk.Frame(params_frame)
        preset_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(preset_frame, text="快速预设:").pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="默认参数", command=self._preset_default).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="保守参数", command=self._preset_conservative).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="敏感参数", command=self._preset_sensitive).pack(side=tk.LEFT, padx=5)
        
    def _create_param_row(self, parent, label_text, variable, tooltip, row, 
                          from_=0, to=100, resolution=1, is_int=False):
        """创建参数行"""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        
        if is_int:
            spinbox = ttk.Spinbox(parent, textvariable=variable, from_=from_, to=to, width=10)
        else:
            spinbox = ttk.Spinbox(parent, textvariable=variable, from_=from_, to=to, 
                                 increment=resolution, width=10)
        spinbox.grid(row=row, column=1, padx=5, pady=5)
        
        # 滑块
        if is_int:
            scale = ttk.Scale(parent, variable=variable, from_=from_, to=to, 
                             orient=tk.HORIZONTAL, length=200)
        else:
            scale = ttk.Scale(parent, variable=variable, from_=from_, to=to,
                             orient=tk.HORIZONTAL, length=200)
        scale.grid(row=row, column=2, padx=5, pady=5)
        
        ttk.Label(parent, text=tooltip, foreground='gray').grid(row=row, column=3, sticky=tk.W, padx=5)
        
    def _create_train_page(self):
        """创建训练页面"""
        train_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(train_frame, text="🚀 训练")
        
        # 训练控制
        control_group = ttk.LabelFrame(train_frame, text="训练控制", padding="10")
        control_group.pack(fill=tk.X, pady=5)
        
        btn_frame = ttk.Frame(control_group)
        btn_frame.pack(fill=tk.X)
        
        self.train_btn = ttk.Button(btn_frame, text="开始训练", command=self._start_training)
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="显示可视化", command=self._show_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="打印摘要", command=self._print_summary).pack(side=tk.LEFT, padx=5)
        
        # 进度条
        self.progress = ttk.Progressbar(control_group, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=10)
        
        self.status_label = ttk.Label(control_group, text="就绪")
        self.status_label.pack(anchor=tk.W)
        
        # 训练日志
        log_group = ttk.LabelFrame(train_frame, text="训练日志", padding="10")
        log_group.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_group, height=15, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # 训练结果
        result_group = ttk.LabelFrame(train_frame, text="训练结果", padding="10")
        result_group.pack(fill=tk.X, pady=5)
        
        self.result_tree = ttk.Treeview(result_group, columns=("cluster", "samples", "ssc", "mean_vol", "mean_dur"),
                                        show="headings", height=5)
        
        self.result_tree.heading("cluster", text="簇ID")
        self.result_tree.heading("samples", text="样本数")
        self.result_tree.heading("ssc", text="SSC规则值")
        self.result_tree.heading("mean_vol", text="平均体积")
        self.result_tree.heading("mean_dur", text="平均时长")
        
        self.result_tree.column("cluster", width=80)
        self.result_tree.column("samples", width=100)
        self.result_tree.column("ssc", width=120)
        self.result_tree.column("mean_vol", width=120)
        self.result_tree.column("mean_dur", width=120)
        
        self.result_tree.pack(fill=tk.X)
        
    def _create_model_page(self):
        """创建模型管理页面"""
        model_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(model_frame, text="💾 模型管理")
        
        # 保存模型
        save_group = ttk.LabelFrame(model_frame, text="保存模型", padding="10")
        save_group.pack(fill=tk.X, pady=5)
        
        save_row = ttk.Frame(save_group)
        save_row.pack(fill=tk.X)
        
        ttk.Label(save_row, text="保存路径:").pack(side=tk.LEFT)
        ttk.Entry(save_row, textvariable=self.model_save_path, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(save_row, text="浏览...", command=self._browse_save_path).pack(side=tk.LEFT)
        ttk.Button(save_row, text="保存模型", command=self._save_model).pack(side=tk.LEFT, padx=5)
        
        # 加载模型
        load_group = ttk.LabelFrame(model_frame, text="加载模型", padding="10")
        load_group.pack(fill=tk.X, pady=5)
        
        ttk.Button(load_group, text="加载已有模型", command=self._load_model).pack(side=tk.LEFT, padx=5)
        
        self.model_info_label = ttk.Label(load_group, text="尚未加载模型")
        self.model_info_label.pack(side=tk.LEFT, padx=20)
        
        # 模型信息
        info_group = ttk.LabelFrame(model_frame, text="当前模型信息", padding="10")
        info_group.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.model_info_text = scrolledtext.ScrolledText(info_group, height=15, state=tk.DISABLED)
        self.model_info_text.pack(fill=tk.BOTH, expand=True)
        
    def _browse_csv(self):
        """浏览CSV文件"""
        filename = filedialog.askopenfilename(
            title="选择CSV文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
            initialdir=os.path.dirname(self.csv_path.get()) if self.csv_path.get() else None
        )
        if filename:
            self.csv_path.set(filename)
            self._load_preview()
            
    def _load_preview(self):
        """加载数据预览"""
        csv_path = self.csv_path.get()
        if not csv_path or not os.path.exists(csv_path):
            messagebox.showerror("错误", "请选择有效的CSV文件")
            return
            
        try:
            # 加载数据
            self.loaded_data = DataLoader.load_csv(
                csv_path,
                volume_col=self.volume_col.get() or None,
                duration_col=self.duration_col.get() or None,
                feeding_id_col=self.feeding_id_col.get() or None
            )
            
            # 更新列选择下拉框
            columns = [''] + self.loaded_data['columns']
            self.volume_combo['values'] = columns
            self.duration_combo['values'] = columns
            self.feeding_combo['values'] = columns
            
            # 清空预览
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            # 填充预览（最多显示100行）
            volumes = self.loaded_data['volumes']
            durations = self.loaded_data['durations']
            feeding_ids = self.loaded_data['feeding_ids']
            
            if volumes is not None and durations is not None:
                n_show = min(100, len(volumes))
                for i in range(n_show):
                    self.data_tree.insert("", tk.END, values=(
                        i + 1,
                        f"{volumes[i]:.4f}" if volumes is not None else "N/A",
                        f"{durations[i]:.4f}" if durations is not None else "N/A",
                        int(feeding_ids[i]) if feeding_ids is not None else "N/A"
                    ))
                
                # 统计信息
                stats_text = f"数据格式: {self.loaded_data['detected_format']} | "
                stats_text += f"总样本数: {len(volumes)} | "
                stats_text += f"体积范围: [{volumes.min():.2f}, {volumes.max():.2f}] | "
                stats_text += f"时长范围: [{durations.min():.2f}, {durations.max():.2f}]"
                self.data_stats_label.config(text=stats_text)
                
                self._log(f"成功加载数据: {csv_path}")
                self._log(f"检测到格式: {self.loaded_data['detected_format']}")
                self._log(f"样本数量: {len(volumes)}")
            else:
                self.data_stats_label.config(text="无法解析数据")
                
        except Exception as e:
            messagebox.showerror("错误", f"加载CSV失败: {str(e)}")
            self._log(f"加载失败: {str(e)}")
            
    def _browse_save_path(self):
        """浏览模型保存路径"""
        filename = filedialog.asksaveasfilename(
            title="选择模型保存位置",
            defaultextension=".pkl",
            filetypes=[("Pickle文件", "*.pkl"), ("所有文件", "*.*")],
            initialdir=os.path.dirname(self.model_save_path.get()) if self.model_save_path.get() else None
        )
        if filename:
            # 移除扩展名，因为save_model会自动添加
            if filename.endswith('.pkl'):
                filename = filename[:-4]
            self.model_save_path.set(filename)
            
    def _preset_default(self):
        """默认参数预设"""
        self.min_cluster_size.set(10)
        self.kde_bandwidth.set(0.5)
        self.T_max.set(5.0)
        self.epsilon.set(0.01)
        self.gamma.set(0.8)
        self.beta.set(1.0)
        self._log("已应用默认参数预设")
        
    def _preset_conservative(self):
        """保守参数预设"""
        self.min_cluster_size.set(20)
        self.kde_bandwidth.set(0.3)
        self.T_max.set(3.0)
        self.epsilon.set(0.01)
        self.gamma.set(0.6)
        self.beta.set(1.5)
        self._log("已应用保守参数预设（适合老年人/患者）")
        
    def _preset_sensitive(self):
        """敏感参数预设"""
        self.min_cluster_size.set(5)
        self.kde_bandwidth.set(0.8)
        self.T_max.set(8.0)
        self.epsilon.set(0.005)
        self.gamma.set(1.0)
        self.beta.set(0.5)
        self._log("已应用敏感参数预设（适合数据量较少或需要细粒度分析）")
        
    def _get_current_config(self) -> TrainingConfig:
        """获取当前配置"""
        return TrainingConfig(
            min_cluster_size=self.min_cluster_size.get(),
            kde_bandwidth=self.kde_bandwidth.get(),
            T_max=self.T_max.get(),
            epsilon=self.epsilon.get(),
            gamma=self.gamma.get(),
            beta=self.beta.get(),
            population=self.population.get()
        )
        
    def _start_training(self):
        """开始训练"""
        if self.loaded_data is None:
            messagebox.showerror("错误", "请先加载数据")
            return
            
        volumes = self.loaded_data['volumes']
        durations = self.loaded_data['durations']
        feeding_ids = self.loaded_data['feeding_ids']
        
        if volumes is None or durations is None:
            messagebox.showerror("错误", "数据无效，无法训练")
            return
        
        # 禁用训练按钮
        self.train_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.status_label.config(text="正在训练...")
        
        # 在后台线程中训练
        def train_thread():
            try:
                config = self._get_current_config()
                self._log(f"开始训练，参数配置: {config.to_dict()}")
                
                # 创建分析器
                self.analyzer = SwallowAnalyzer.from_config(config)
                
                # 预处理数据
                if self.preprocess_data.get():
                    vols, durs, fids = DataLoader.preprocess_data(volumes, durations, feeding_ids)
                    self._log(f"预处理后样本数: {len(vols)} (原始: {len(volumes)})")
                else:
                    vols, durs, fids = volumes, durations, feeding_ids
                
                # 训练
                self.training_result = self.analyzer.train(vols, durs, fids, config.population)
                
                # 更新UI（在主线程中）
                self.root.after(0, self._on_training_complete)
                
            except Exception as e:
                self.root.after(0, lambda: self._on_training_error(str(e)))
        
        threading.Thread(target=train_thread, daemon=True).start()
        
    def _on_training_complete(self):
        """训练完成回调"""
        self.progress.stop()
        self.train_btn.config(state=tk.NORMAL)
        
        if self.training_result and self.training_result.success:
            self.status_label.config(text=f"训练完成: {self.training_result.message}")
            self._log(f"训练成功！")
            self._log(f"发现 {self.training_result.cluster_count} 个聚类簇")
            self._log(f"训练耗时: {self.training_result.training_time:.2f} 秒")
            
            # 更新结果表格
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)
                
            for cluster_id, info in self.training_result.summary.get('clusters', {}).items():
                self.result_tree.insert("", tk.END, values=(
                    f"簇 {cluster_id}",
                    info['sample_count'],
                    f"{info['SSC_rule']:.2f} mL",
                    f"{info['mean_volume']:.2f} mL",
                    f"{info['mean_duration']:.2f} s"
                ))
                
            # 更新模型信息
            self._update_model_info()
            
            messagebox.showinfo("成功", self.training_result.message)
        else:
            msg = self.training_result.message if self.training_result else "未知错误"
            self.status_label.config(text=f"训练失败: {msg}")
            self._log(f"训练失败: {msg}")
            messagebox.showerror("失败", msg)
            
    def _on_training_error(self, error_msg: str):
        """训练错误回调"""
        self.progress.stop()
        self.train_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"训练出错: {error_msg}")
        self._log(f"训练出错: {error_msg}")
        messagebox.showerror("错误", f"训练过程中出错:\n{error_msg}")
        
    def _show_visualization(self):
        """显示可视化"""
        if self.analyzer is None or not self.analyzer.is_trained():
            messagebox.showerror("错误", "请先完成训练")
            return
            
        try:
            results = self.analyzer.get_results()
            if results:
                self.analyzer.visualize_results(results)
            else:
                messagebox.showerror("错误", "没有可用的训练结果")
        except Exception as e:
            messagebox.showerror("错误", f"显示可视化失败: {str(e)}")
            
    def _print_summary(self):
        """打印摘要"""
        if self.analyzer is None or not self.analyzer.is_trained():
            messagebox.showerror("错误", "请先完成训练")
            return
            
        try:
            results = self.analyzer.get_results()
            if results:
                self.analyzer.print_summary(results)
                self._log("摘要已打印到控制台")
            else:
                messagebox.showerror("错误", "没有可用的训练结果")
        except Exception as e:
            messagebox.showerror("错误", f"打印摘要失败: {str(e)}")
            
    def _save_model(self):
        """保存模型"""
        if self.analyzer is None or not self.analyzer.is_trained():
            messagebox.showerror("错误", "请先完成训练")
            return
            
        save_path = self.model_save_path.get()
        if not save_path:
            messagebox.showerror("错误", "请指定保存路径")
            return
            
        try:
            pkl_path = self.analyzer.save_model(
                save_path, 
                include_data=self.include_data_in_model.get()
            )
            self._log(f"模型已保存到: {pkl_path}")
            messagebox.showinfo("成功", f"模型已保存到:\n{pkl_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存模型失败: {str(e)}")
            
    def _load_model(self):
        """加载模型"""
        filename = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("Pickle文件", "*.pkl"), ("所有文件", "*.*")]
        )
        if not filename:
            return
            
        try:
            self.analyzer = SwallowAnalyzer.load_model(filename)
            self._log(f"已加载模型: {filename}")
            
            # 更新配置显示
            config = self.analyzer.get_config()
            self.min_cluster_size.set(config.min_cluster_size)
            self.kde_bandwidth.set(config.kde_bandwidth)
            self.T_max.set(config.T_max)
            self.epsilon.set(config.epsilon)
            self.gamma.set(config.gamma)
            self.beta.set(config.beta)
            # 回填已保存的模型中记录的人群类型到界面（normal/elderly/patient）
            try:
                self.population.set(config.population)
            except Exception:
                pass
            
            self.model_info_label.config(text=f"已加载: {os.path.basename(filename)}")
            self._update_model_info()
            
            messagebox.showinfo("成功", "模型加载成功")
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {str(e)}")
            
    def _update_model_info(self):
        """更新模型信息显示"""
        self.model_info_text.config(state=tk.NORMAL)
        self.model_info_text.delete(1.0, tk.END)
        
        if self.analyzer is None:
            self.model_info_text.insert(tk.END, "尚未加载或训练模型")
        else:
            config = self.analyzer.get_config()
            info_lines = [
                "=" * 50,
                "当前模型配置",
                "=" * 50,
                f"最小簇大小: {config.min_cluster_size}",
                f"KDE带宽: {config.kde_bandwidth}",
                f"最大吞咽时间: {config.T_max} 秒",
                f"Epsilon: {config.epsilon}",
                f"Gamma: {config.gamma}",
                f"Beta: {config.beta}",
                "",
            ]
            
            results = self.analyzer.get_results()
            if results:
                info_lines.extend([
                    "=" * 50,
                    "聚类结果",
                    "=" * 50,
                ])
                
                cluster_info = results.get('cluster_info', {})
                for k in sorted(cluster_info.keys()):
                    info = cluster_info[k]
                    v_max, t_max = info['max_density_point']
                    info_lines.extend([
                        f"\n[簇 {k}]",
                        f"  样本数量: {info['sample_count']}",
                        f"  峰值密度点: 体积={v_max:.2f} mL, 时长={t_max:.2f} s",
                        f"  SSC规则值: {info['SSC_rule']:.2f} mL",
                        f"  平均体积: {info['mean_volume']:.2f} mL",
                        f"  平均时长: {info['mean_duration']:.2f} s",
                    ])
                    
            self.model_info_text.insert(tk.END, "\n".join(info_lines))
            
        self.model_info_text.config(state=tk.DISABLED)
        
    def _log(self, message: str):
        """添加日志"""
        self.log_text.config(state=tk.NORMAL)
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)


def main():
    """主函数"""
    root = tk.Tk()
    
    # 设置样式
    style = ttk.Style()
    style.theme_use('clam')  # 使用clam主题，在Windows上也比较美观
    
    app = TrainUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
