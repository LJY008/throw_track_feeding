# QV实时监测器 - 三线程架构说明

## 架构概览

```
neck_throat_track.py (ZMQ Publisher)
         ↓ (tcp://localhost:5555, msgpack)
    ┌────────────────────────────────────┐
    │  qv_realtime_monitor.py            │
    │                                    │
    │  Thread 1: ZMQ接收线程             │
    │    - 接收数据并放入data_queue      │
    │         ↓                          │
    │  Thread 2: 检测处理线程            │
    │    - 从data_queue取数据            │
    │    - 缓冲到300个样本               │
    │    - 运行QV检测                    │
    │    - 结果放入result_queue          │
    │         ↓                          │
    │  Thread 3 (主线程): GUI更新        │
    │    - 定时从result_queue取结果      │
    │    - 更新matplotlib图表            │
    │    - 更新统计信息                  │
    └────────────────────────────────────┘
```

## 主要改动

### 1. 完全移除ROS2
- 删除所有ROS2相关代码
- 使用ZeroMQ替代（更轻量，跨平台）

### 2. 三线程并行架构
- **线程1（ZMQ-Receiver）**: 独立接收数据
- **线程2（Detection-Processor）**: 独立处理检测
- **线程3（GUI主线程）**: 定时更新图表（100ms）

### 3. 工作模式选择
- **数据源模式**：
  - 实时监测（默认）：从ZMQ接收数据
  - CSV离线分析：从文件加载数据
  
- **检测轴模式**：
  - 双轴（默认）：Y+Z平均
  - 单轴：仅Y或仅Z

### 4. 操作流程

#### 实时监测模式：
1. 启动程序（默认实时模式）
2. 选择检测轴（双轴/单轴）
3. 点击 "▶ 开始实时监测"
4. 三线程自动启动，图表实时更新
5. 点击 "⏹ 停止监测" 停止

#### CSV离线分析模式：
1. 切换到 "CSV离线分析" 模式
2. 点击 "加载CSV文件"
3. 选择检测轴（双轴/单轴）
4. 点击 "▶ 运行CSV检测"
5. 一次性完成检测并显示结果

### 5. 模式切换规则
- **监测前**：可自由切换数据源模式和检测轴模式
- **监测中**：模式锁定，需先停止监测才能切换

## 文件说明

### qv_realtime_monitor.py（已重构）
- **RealtimeDataQueue**: 线程安全队列（maxsize=1000）
- **ZMQRealtimeInterface**: ZMQ接收线程
- **DetectionProcessor**: 检测处理线程（buffer_size=300）
- **QVMonitorGUI**: GUI主类
  - `start_monitoring()`: 启动三线程
  - `stop_monitoring()`: 停止三线程
  - `schedule_gui_update()`: 定时GUI更新（100ms周期）

### neck_throat_track.py（已有ZMQ实现）
- 已实现ZMQ Publisher
- 端口: 5555
- 数据格式: msgpack序列化
```python
{
    'timestamp': float,
    'track_id': int,
    'y_displacement': float,
    'depth_displacement': float,
    'neck_origin': tuple,
    'body_motion_detected': bool,
    'current_volatility': float
}
```

## 运行方式

### 启动顺序：
```bash
# 终端1：启动数据发布端
cd e:\nc-code
python neck_throat_track.py

# 终端2：启动监测GUI
cd e:\nc-code
python qv_realtime_monitor.py
```

### 验证测试：
```bash
# 运行单元测试
cd e:\nc-code
python test_qv_monitor.py
```

## 配置管理

### 保存/加载配置
- 配置文件格式：JSON
- 包含内容：
  - 工作模式（is_realtime_mode, is_dual_axis）
  - 检测参数（fs, h, H, H2等）
  - 卡尔曼参数（kalman_Q, kalman_R_factor）
  - 评分权重（w1, w2, w3）
  - 置零阈值（zero_threshold）

### 按钮功能：
- **重置配置**: 恢复默认值
- **保存配置**: 弹窗选择保存位置
- **导出配置**: 导出到 `qv_monitor_config_export.json`
- **加载配置**: 从文件加载（监测中禁用）

## 性能参数

- **接收队列**: 1000条消息（溢出自动丢弃旧数据）
- **结果队列**: 10条结果（溢出自动丢弃旧结果）
- **检测缓冲**: 300个样本（约10秒@30Hz）
- **GUI更新**: 100ms周期（10 FPS）
- **ZMQ Socket**: CONFLATE模式（只保留最新消息）

## 线程安全

### 数据传递方式：
- ZMQ线程 → 检测线程: `queue.Queue` (data_queue)
- 检测线程 → GUI线程: `queue.Queue` (result_queue)
- GUI更新: `root.after()` 定时器（线程安全）

### 避免竞态条件：
- 使用`queue.Queue`（内置线程安全）
- 使用`threading.Event`控制线程停止
- matplotlib更新仅在主线程执行

## 故障排除

### 问题1：ZMQ连接失败
- 检查neck_throat_track是否运行
- 确认端口5555未被占用
- 查看控制台ZMQ日志

### 问题2：检测无结果
- 确认缓冲区已满（需300个样本）
- 检查数据格式是否正确
- 查看检测参数设置

### 问题3：图表不更新
- 确认点击了 "开始实时监测"
- 检查result_queue是否有数据
- 查看控制台错误信息

## 日志输出

### 正常启动日志：
```
✅ ZMQ接收线程已启动
✅ 检测处理线程已启动
✅ 实时监测已启动
   - 检测模式: 双轴
```

### 运行中日志：
```
📊 已接收 1000 条消息，队列大小: 5
```

### 停止日志：
```
✅ 监测已停止
   - 总接收消息数: 2450
```

## 后续优化方向

1. **性能优化**：
   - 可调节GUI更新频率
   - 可调节检测缓冲区大小
   - 添加性能监控面板

2. **功能增强**：
   - 添加数据录制功能
   - 添加事件导出功能
   - 添加多个检测器并行

3. **可视化改进**：
   - 添加实时波形滚动显示
   - 添加事件历史记录
   - 添加统计图表

## 开发者备注

- 代码已移除所有ROS2依赖
- 所有GUI组件线程安全
- 配置文件向后兼容旧格式
- 测试覆盖所有主要组件

---

**版本**: v2.0 (三线程架构)  
**日期**: 2025-01-20  
**作者**: GitHub Copilot  
