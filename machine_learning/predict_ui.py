"""
SSC预测系统 - UI界面
加载训练好的模型，根据上次喂食量和吞咽时间生成下一个喂食量推荐
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import sys
import os
from datetime import datetime
from typing import Optional, Dict, List
import json

# 确保可以导入同目录的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import SwallowAnalyzer, TrainingConfig


class FeedingHistory:
    """喂食历史记录管理"""
    
    def __init__(self, max_records: int = 100):
        self.max_records = max_records
        self.records: List[Dict] = []
        
    def add_record(self, volume: float, duration: float, 
                   recommended_ssc: float, cluster_id: int, 
                   population: str, timestamp: str = None):
        """添加一条喂食记录"""
        record = {
            'timestamp': timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'volume': volume,
            'duration': duration,
            'recommended_ssc': recommended_ssc,
            'cluster_id': cluster_id,
            'population': population
        }
        self.records.append(record)
        
        # 保持记录数量限制
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]
            
    def get_last_record(self) -> Optional[Dict]:
        """获取最近一条记录"""
        return self.records[-1] if self.records else None
    
    def get_records(self, n: int = 10) -> List[Dict]:
        """获取最近n条记录"""
        return self.records[-n:] if self.records else []
    
    def clear(self):
        """清空历史记录"""
        self.records = []
        
    def save_to_file(self, file_path: str):
        """保存历史记录到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)
            
    def load_from_file(self, file_path: str):
        """从文件加载历史记录"""
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.records = json.load(f)


class PredictUI:
    """SSC预测系统UI界面"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SSC预测系统 - 喂食量推荐")
        self.root.geometry("800x650")
        self.root.minsize(700, 550)
        
        # 初始化变量
        self.model_path = tk.StringVar()
        self.input_volume = tk.DoubleVar(value=3.0)
        self.input_duration = tk.DoubleVar(value=2.0)
        self.population = tk.StringVar(value='normal')
        self.apply_adjustment = tk.BooleanVar(value=True)
        
        # 模型和历史记录
        self.analyzer: Optional[SwallowAnalyzer] = None
        self.history = FeedingHistory()
        
        # 历史记录文件路径
        self.history_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'feeding_history.json'
        )
        
        # 尝试加载历史记录
        try:
            self.history.load_from_file(self.history_file)
        except:
            pass
        
        # 创建UI
        self._create_ui()
        
    def _create_ui(self):
        """创建UI布局"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 模型加载区域
        self._create_model_section(main_frame)
        
        # 输入区域
        self._create_input_section(main_frame)
        
        # 预测结果区域
        self._create_result_section(main_frame)
        
        # 历史记录区域
        self._create_history_section(main_frame)
        
    def _create_model_section(self, parent):
        """创建模型加载区域"""
        model_frame = ttk.LabelFrame(parent, text="模型加载", padding="10")
        model_frame.pack(fill=tk.X, pady=5)
        
        row1 = ttk.Frame(model_frame)
        row1.pack(fill=tk.X)
        
        ttk.Label(row1, text="模型文件:").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.model_path, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(row1, text="浏览...", command=self._browse_model).pack(side=tk.LEFT)
        ttk.Button(row1, text="加载模型", command=self._load_model).pack(side=tk.LEFT, padx=5)
        
        # 模型信息标签
        self.model_info_label = ttk.Label(model_frame, text="尚未加载模型", foreground='gray')
        self.model_info_label.pack(anchor=tk.W, pady=5)
        
    def _create_input_section(self, parent):
        """创建输入区域"""
        input_frame = ttk.LabelFrame(parent, text="输入参数", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        # 输入网格
        grid_frame = ttk.Frame(input_frame)
        grid_frame.pack(fill=tk.X)
        
        # 喂食量输入
        ttk.Label(grid_frame, text="上次喂食量 (mL):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        volume_frame = ttk.Frame(grid_frame)
        volume_frame.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.volume_spinbox = ttk.Spinbox(volume_frame, textvariable=self.input_volume, 
                                          from_=0.1, to=20.0, increment=0.1, width=10)
        self.volume_spinbox.pack(side=tk.LEFT)
        
        self.volume_scale = ttk.Scale(volume_frame, variable=self.input_volume, 
                                      from_=0.1, to=20.0, orient=tk.HORIZONTAL, length=150)
        self.volume_scale.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(grid_frame, text="范围: 0.1 - 20.0 mL", foreground='gray').grid(row=0, column=2, sticky=tk.W)
        
        # 吞咽时间输入
        ttk.Label(grid_frame, text="上次吞咽时间 (秒):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        duration_frame = ttk.Frame(grid_frame)
        duration_frame.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.duration_spinbox = ttk.Spinbox(duration_frame, textvariable=self.input_duration,
                                            from_=0.1, to=10.0, increment=0.1, width=10)
        self.duration_spinbox.pack(side=tk.LEFT)
        
        self.duration_scale = ttk.Scale(duration_frame, variable=self.input_duration,
                                        from_=0.1, to=10.0, orient=tk.HORIZONTAL, length=150)
        self.duration_scale.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(grid_frame, text="范围: 0.1 - 10.0 秒", foreground='gray').grid(row=1, column=2, sticky=tk.W)
        
        # 人群类型选择
        ttk.Label(grid_frame, text="人群类型:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        pop_frame = ttk.Frame(grid_frame)
        pop_frame.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        for pop_val, pop_text in [('normal', '正常人'), ('elderly', '老年人'), ('patient', '患者')]:
            ttk.Radiobutton(pop_frame, text=pop_text, variable=self.population, 
                           value=pop_val).pack(side=tk.LEFT, padx=10)
        
        # 是否应用人群调整
        ttk.Checkbutton(grid_frame, text="应用人群安全系数调整", 
                       variable=self.apply_adjustment).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # 预测按钮
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.predict_btn = ttk.Button(btn_frame, text="🔮 生成喂食量推荐", 
                                      command=self._predict, style='Accent.TButton')
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="使用上次记录", command=self._use_last_record).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="重置输入", command=self._reset_input).pack(side=tk.LEFT, padx=5)
        
    def _create_result_section(self, parent):
        """创建预测结果区域"""
        result_frame = ttk.LabelFrame(parent, text="推荐结果", padding="10")
        result_frame.pack(fill=tk.X, pady=5)
        
        # 主推荐结果（大字体显示）
        self.main_result_frame = ttk.Frame(result_frame)
        self.main_result_frame.pack(fill=tk.X, pady=5)
        
        self.result_label = ttk.Label(self.main_result_frame, 
                                      text="请先加载模型并输入参数",
                                      font=('Arial', 16, 'bold'))
        self.result_label.pack()
        
        # 详细结果
        detail_frame = ttk.Frame(result_frame)
        detail_frame.pack(fill=tk.X, pady=5)
        
        # 结果表格
        columns = ("item", "value")
        self.result_tree = ttk.Treeview(detail_frame, columns=columns, show="headings", height=6)
        self.result_tree.heading("item", text="指标")
        self.result_tree.heading("value", text="值")
        self.result_tree.column("item", width=200)
        self.result_tree.column("value", width=300)
        self.result_tree.pack(fill=tk.X)
        
    def _create_history_section(self, parent):
        """创建历史记录区域"""
        history_frame = ttk.LabelFrame(parent, text="喂食历史记录", padding="10")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 历史记录表格
        columns = ("time", "volume", "duration", "recommended", "cluster", "population")
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=5)
        
        self.history_tree.heading("time", text="时间")
        self.history_tree.heading("volume", text="喂食量(mL)")
        self.history_tree.heading("duration", text="吞咽时间(s)")
        self.history_tree.heading("recommended", text="推荐量(mL)")
        self.history_tree.heading("cluster", text="簇ID")
        self.history_tree.heading("population", text="人群类型")
        
        self.history_tree.column("time", width=130)
        self.history_tree.column("volume", width=90)
        self.history_tree.column("duration", width=90)
        self.history_tree.column("recommended", width=90)
        self.history_tree.column("cluster", width=60)
        self.history_tree.column("population", width=80)
        
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 历史记录操作按钮
        btn_frame = ttk.Frame(history_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="保存历史", command=self._save_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="清空历史", command=self._clear_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="导出CSV", command=self._export_history_csv).pack(side=tk.LEFT, padx=5)
        
        # 初始化历史记录显示
        self._update_history_display()
        
    def _browse_model(self):
        """浏览模型文件"""
        filename = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("Pickle文件", "*.pkl"), ("所有文件", "*.*")],
            initialdir=os.path.dirname(self.model_path.get()) if self.model_path.get() else None
        )
        if filename:
            self.model_path.set(filename)
            self._load_model()
            
    def _load_model(self):
        """加载模型"""
        model_path = self.model_path.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("错误", "请选择有效的模型文件")
            return
            
        try:
            self.analyzer = SwallowAnalyzer.load_model(model_path)
            
            # 获取模型信息
            config = self.analyzer.get_config()
            results = self.analyzer.get_results()
            
            cluster_count = len(results.get('cluster_info', {})) if results else 0
            
            info_text = f"✅ 模型已加载 | 聚类数: {cluster_count} | "
            info_text += f"T_max: {config.T_max}s | Gamma: {config.gamma}"
            
            self.model_info_label.config(text=info_text, foreground='green')
            self.result_label.config(text="模型已加载，请输入参数进行预测")
            
            messagebox.showinfo("成功", f"模型加载成功！\n发现 {cluster_count} 个聚类簇")
            
        except Exception as e:
            self.model_info_label.config(text=f"❌ 加载失败: {str(e)}", foreground='red')
            messagebox.showerror("错误", f"加载模型失败:\n{str(e)}")
            
    def _predict(self):
        """执行预测"""
        if self.analyzer is None or not self.analyzer.is_trained():
            messagebox.showerror("错误", "请先加载训练好的模型")
            return
            
        try:
            volume = self.input_volume.get()
            duration = self.input_duration.get()
            population = self.population.get()
            
            if volume <= 0 or duration <= 0:
                messagebox.showerror("错误", "喂食量和吞咽时间必须大于0")
                return
            
            results = self.analyzer.get_results()
            cluster_info = results.get('cluster_info', {})
            
            if not cluster_info:
                messagebox.showerror("错误", "模型中没有有效的聚类信息")
                return
            
            # 1. 估算该样本的密度并找到最佳匹配簇
            d_new, best_cluster, all_densities = self.analyzer.estimate_density_for_sample(
                volume, duration, cluster_info
            )
            
            if best_cluster is None:
                messagebox.showerror("错误", "无法匹配到任何聚类簇")
                return
            
            # 2. 获取参考密度
            d_ref = cluster_info[best_cluster]['d_ref']
            
            # 3. 计算个体化SSC推荐
            personalized = self.analyzer.compute_personalized_ssc(
                volume, duration, best_cluster, results, 
                percentile=75, 
                apply_population_factor=self.apply_adjustment.get()
            )
            
            # 4. 计算调整后的SSC
            base_ssc = personalized['ssc_recommended']
            adjusted = self.analyzer.compute_adjusted_ssc(base_ssc, d_new, d_ref, population)
            
            # 5. 最终推荐值
            if self.apply_adjustment.get():
                final_ssc = adjusted['ssc_adjusted']
            else:
                final_ssc = base_ssc
            
            # 更新结果显示
            self._display_result(volume, duration, final_ssc, best_cluster, 
                                personalized, adjusted, population)
            
            # 添加到历史记录
            self.history.add_record(
                volume=volume,
                duration=duration,
                recommended_ssc=final_ssc,
                cluster_id=best_cluster,
                population=population
            )
            self._update_history_display()
            
        except Exception as e:
            messagebox.showerror("错误", f"预测失败:\n{str(e)}")
            import traceback
            traceback.print_exc()
            
    def _display_result(self, volume: float, duration: float, final_ssc: float,
                       cluster_id: int, personalized: Dict, adjusted: Dict, population: str):
        """显示预测结果"""
        # 主结果
        pop_names = {'normal': '正常人', 'elderly': '老年人', 'patient': '患者'}
        self.result_label.config(
            text=f"推荐下次喂食量: {final_ssc:.2f} mL",
            foreground='#007ACC'
        )
        
        # 清空详细结果表格
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        
        # 填充详细结果
        details = [
            ("输入喂食量", f"{volume:.2f} mL"),
            ("输入吞咽时间", f"{duration:.2f} 秒"),
            ("匹配簇ID", f"簇 {cluster_id}"),
            ("人群类型", pop_names.get(population, population)),
            ("簇基准SSC (SSC_rule)", f"{personalized.get('cluster_ssc_rule', 0):.2f} mL"),
            ("簇平均体积", f"{personalized.get('cluster_vol_mean', 0):.2f} mL"),
            ("密度比率", f"{personalized.get('density_ratio', 0):.2%}"),
            ("基础SSC (分位数)", f"{personalized.get('ssc_percentile_based', 0):.2f} mL"),
            ("密度调整SSC", f"{personalized.get('ssc_density_adjusted', 0):.2f} mL"),
            ("时间调整SSC", f"{personalized.get('ssc_time_adjusted', 0):.2f} mL"),
            ("安全系数", f"{personalized.get('safety_factor', 0):.3f}"),
            ("调整因子 α_new", f"{adjusted.get('alpha_new', 0):.4f}"),
            ("最终调整系数 γ", f"{adjusted.get('gamma_final', 0):.4f}"),
            ("最终推荐量", f"{final_ssc:.2f} mL"),
        ]
        
        for item, value in details:
            self.result_tree.insert("", tk.END, values=(item, value))
            
    def _use_last_record(self):
        """使用上一次记录的数据"""
        last_record = self.history.get_last_record()
        if last_record:
            self.input_volume.set(last_record['volume'])
            self.input_duration.set(last_record['duration'])
            self.population.set(last_record.get('population', 'normal'))
        else:
            messagebox.showinfo("提示", "没有历史记录")
            
    def _reset_input(self):
        """重置输入"""
        self.input_volume.set(3.0)
        self.input_duration.set(2.0)
        self.population.set('normal')
        
    def _update_history_display(self):
        """更新历史记录显示"""
        # 清空表格
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # 填充历史记录（最新的在前）
        pop_names = {'normal': '正常人', 'elderly': '老年人', 'patient': '患者'}
        for record in reversed(self.history.get_records(50)):
            self.history_tree.insert("", tk.END, values=(
                record.get('timestamp', ''),
                f"{record.get('volume', 0):.2f}",
                f"{record.get('duration', 0):.2f}",
                f"{record.get('recommended_ssc', 0):.2f}",
                record.get('cluster_id', ''),
                pop_names.get(record.get('population', ''), record.get('population', ''))
            ))
            
    def _save_history(self):
        """保存历史记录"""
        try:
            self.history.save_to_file(self.history_file)
            messagebox.showinfo("成功", f"历史记录已保存到:\n{self.history_file}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败:\n{str(e)}")
            
    def _clear_history(self):
        """清空历史记录"""
        if messagebox.askyesno("确认", "确定要清空所有历史记录吗？"):
            self.history.clear()
            self._update_history_display()
            
    def _export_history_csv(self):
        """导出历史记录为CSV"""
        if not self.history.records:
            messagebox.showinfo("提示", "没有历史记录可导出")
            return
            
        filename = filedialog.asksaveasfilename(
            title="导出历史记录",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if filename:
            try:
                import csv
                with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        'timestamp', 'volume', 'duration', 
                        'recommended_ssc', 'cluster_id', 'population'
                    ])
                    writer.writeheader()
                    writer.writerows(self.history.records)
                messagebox.showinfo("成功", f"历史记录已导出到:\n{filename}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败:\n{str(e)}")


def main():
    """主函数"""
    root = tk.Tk()
    
    # 设置样式
    style = ttk.Style()
    style.theme_use('clam')
    
    app = PredictUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
