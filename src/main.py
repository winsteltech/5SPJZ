#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5通道圆形电容感应上位机 - 主窗口
布局：第一行左热力图+右柱状图，第二行曲线图
"""

import sys
import os
from pathlib import Path
import time
import math
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QMessageBox, QFileDialog, QGroupBox, QFormLayout, QCheckBox, QTextEdit
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont

import pyqtgraph as pg
import numpy as np

from src.communication.serial_communicator import SerialCommunicator
from src.processing.signal_processor import StreamingFilter
from src.storage.database_manager import DatabaseManager
from src.visualization.realtime_visualizer import ForceField3DWidget, BarChartWidget, CurveVisualizer
from src.api.data_api_server import DataAPIServer


NUM_CHANNELS = 5


class DataReceiverThread(QThread):
    """数据接收线程"""
    data_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, communicator: SerialCommunicator):
        super().__init__()
        self.communicator = communicator
        self.is_running = True
        self.data_counter = 0

    def run(self):
        while self.is_running:
            if self.communicator.is_connected:
                data = self.communicator.get_data(timeout=0.1)
                if data:
                    self.data_counter += 1
                    self.data_received.emit(data)
            time.sleep(0.01)

    def stop(self):
        self.is_running = False


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('5通道圆形电容感应数据可视化系统 v1.6')
        self.setGeometry(100, 100, 1400, 900)

        # 初始化组件
        self.communicator = None
        self.receiver_thread = None
        self.db_manager = DatabaseManager()
        self.streaming_filters = {i: StreamingFilter() for i in range(NUM_CHANNELS)}

        # 数据归零
        self.initial_values = {ch: 0.0 for ch in range(NUM_CHANNELS)}
        self.use_initial_zero = True
        self.channel_flip_states = {ch: False for ch in range(NUM_CHANNELS)}
        self.latest_raw_data = {}

        # 初始数据收集
        self.connection_time = None
        self.is_collecting_initial = False
        self.initial_data_collector = {ch: [] for ch in range(NUM_CHANNELS)}
        self.initialization_done = False

        # 统计
        self.packet_count = 0
        self.error_count = 0

        # 心跳检测
        self.last_data_time = 0
        self.heartbeat_timeout = 3.0

        # 无压力自动归零机制
        self.no_pressure_start_time = None
        self.no_pressure_timeout = 2.0  # 超过2秒无压力信号则自动归零
        self.pressure_threshold = 30.0   # 压力判定阈值（低于此值视为无压力）
        self.idle_rezero_count = 0        # 空闲归零计数器

        # 迟滞补偿：上一帧处理后的数据
        self._prev_processed = None

        # API服务器
        self.api_server = DataAPIServer(host='0.0.0.0', port=5000)
        self.api_enabled = True

        # 3D受力场镜像翻转状态
        self.force_mirror_h = False  # 水平翻转
        self.force_mirror_v = False  # 垂直翻转

        # 初始化UI
        self.init_ui()

        # 串口自动检测
        self.port_detection_timer = QTimer(self)
        self.port_detection_timer.timeout.connect(self.update_port_list)
        self.port_detection_timer.start(1000)

        # 界面更新
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(100)

        # 心跳检测
        self.heartbeat_timer = QTimer(self)
        self.heartbeat_timer.timeout.connect(self.check_heartbeat)
        self.heartbeat_timer.start(500)

    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()

        # 左侧：控制面板
        left_widget = self.create_control_panel()
        main_layout.addWidget(left_widget, 1)

        # 右侧：可视化区域
        right_widget = self.create_visualization_panel()
        main_layout.addWidget(right_widget, 3)

        central_widget.setLayout(main_layout)

    def create_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout()

        title = QLabel('控制中心')
        title.setFont(QFont('Arial', 12, QFont.Bold))
        layout.addWidget(title)

        # 选项卡
        self.tabs = QTabWidget()
        tab1 = self.create_serial_config_tab()
        self.tabs.addTab(tab1, '串口配置')
        tab2 = self.create_processing_tab()
        self.tabs.addTab(tab2, '数据处理')
        tab3 = self.create_stats_tab()
        self.tabs.addTab(tab3, '统计')
        layout.addWidget(self.tabs)

        # 按钮组
        button_layout = QVBoxLayout()

        self.connect_btn = QPushButton('连接串口')
        self.connect_btn.clicked.connect(self.on_connect)
        button_layout.addWidget(self.connect_btn)

        init_btn = QPushButton('初始化归零')
        init_btn.clicked.connect(self.on_initialize_zero)
        button_layout.addWidget(init_btn)

        flip_btn = QPushButton('变量翻转')
        flip_btn.clicked.connect(self.on_toggle_flip)
        button_layout.addWidget(flip_btn)

        save_btn = QPushButton('保存数据')
        save_btn.clicked.connect(self.on_save_data)
        button_layout.addWidget(save_btn)

        export_btn = QPushButton('导出 CSV')
        export_btn.clicked.connect(self.on_export_csv)
        button_layout.addWidget(export_btn)

        clear_btn = QPushButton('清空数据')
        clear_btn.clicked.connect(self.on_clear_data)
        button_layout.addWidget(clear_btn)

        about_btn = QPushButton('关于')
        about_btn.clicked.connect(self.on_about)
        button_layout.addWidget(about_btn)

        layout.addLayout(button_layout)
        layout.addStretch()

        panel.setLayout(layout)
        return panel

    def create_serial_config_tab(self) -> QWidget:
        """创建串口配置选项卡"""
        widget = QWidget()
        layout = QFormLayout()

        self.port_combo = QComboBox()
        self.port_combo.addItems(self.get_available_ports())
        layout.addRow('串口:', self.port_combo)

        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(['9600', '115200', '230400', '460800'])
        self.baudrate_combo.setCurrentText('115200')
        self.baudrate_combo.setEditable(True)
        layout.addRow('波特率:', self.baudrate_combo)

        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setValue(1.0)
        self.timeout_spin.setMinimum(0.1)
        layout.addRow('超时(s):', self.timeout_spin)

        self.status_label = QLabel('未连接')
        self.status_label.setStyleSheet('color: red')
        layout.addRow('状态:', self.status_label)

        widget.setLayout(layout)
        return widget

    def create_processing_tab(self) -> QWidget:
        """创建数据处理选项卡"""
        widget = QWidget()
        layout = QVBoxLayout()

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel('滤波算法:'))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(['无', '移动平均', '指数平滑', '卡尔曼滤波', '中值滤波'])
        self.filter_combo.setCurrentIndex(1)
        filter_layout.addWidget(self.filter_combo)
        layout.addLayout(filter_layout)

        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel('窗口大小:'))
        self.window_spin = QSpinBox()
        self.window_spin.setValue(10)
        self.window_spin.setMinimum(1)
        self.window_spin.setMaximum(100)
        window_layout.addWidget(self.window_spin)
        layout.addLayout(window_layout)

        widget.setLayout(layout)
        return widget

    def create_stats_tab(self) -> QWidget:
        """创建统计信息选项卡"""
        widget = QWidget()
        layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont('Courier', 9))
        layout.addWidget(self.stats_text)

        widget.setLayout(layout)
        return widget

    def create_visualization_panel(self) -> QWidget:
        """创建可视化面板
        第一行：左3D受力场 + 右柱状图
        第二行：曲线图
        """
        panel = QWidget()
        main_layout = QVBoxLayout()

        # 第一行：3D受力场 + 柱状图
        row1_widget = QWidget()
        row1_layout = QHBoxLayout()

        # 左侧：3D受力场可视化（内含迷你热力图）
        force_group = QGroupBox('3D受力场分析')
        force_layout = QVBoxLayout()
        self.force_field_widget = ForceField3DWidget()
        force_layout.addWidget(self.force_field_widget)

        # 镜像翻转按钮
        mirror_btn = QPushButton('镜像翻转')
        mirror_btn.setToolTip('点击翻转3D受力场显示（上下+左右）')
        mirror_btn.clicked.connect(self.on_mirror_flip)
        force_layout.addWidget(mirror_btn)

        force_group.setLayout(force_layout)
        row1_layout.addWidget(force_group, 2)

        # 右侧：力学分析柱状图
        bar_group = QGroupBox('力学分析')
        bar_layout = QVBoxLayout()
        self.bar_chart = BarChartWidget()
        bar_layout.addWidget(self.bar_chart)
        bar_group.setLayout(bar_layout)
        row1_layout.addWidget(bar_group, 1)

        row1_widget.setLayout(row1_layout)
        main_layout.addWidget(row1_widget, 3)

        # 第二行：曲线图
        curve_group = QGroupBox('实时曲线')
        curve_layout = QVBoxLayout()

        # 曲线图工具栏：X轴时间窗口设置
        curve_toolbar = QHBoxLayout()
        curve_toolbar.addWidget(QLabel('X轴窗口(秒):'))
        self.x_window_spin = QSpinBox()
        self.x_window_spin.setRange(1, 120)
        self.x_window_spin.setValue(10)
        self.x_window_spin.setSuffix(' s')
        self.x_window_spin.setFixedWidth(80)
        self.x_window_spin.valueChanged.connect(self.on_x_window_changed)
        curve_toolbar.addWidget(self.x_window_spin)
        curve_toolbar.addStretch()
        curve_layout.addLayout(curve_toolbar)

        self.curve_widget = pg.GraphicsLayoutWidget()
        self.curve = CurveVisualizer(channels=NUM_CHANNELS, max_points=1000, x_window=10.0)
        self.curve_plot = self.curve_widget.addPlot(0, 0)
        self.curve_plot.setLabel('bottom', '时间', units='s')
        self.curve_plot.setLabel('left', '电容值')
        self.curve_plot.addLegend()
        self.curve_plot.showGrid(x=True, y=True)

        self.curve_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.curve_widget.customContextMenuRequested.connect(self.show_curve_context_menu)

        for ch in range(NUM_CHANNELS):
            self.curve.add_plot(ch, self.curve_plot)
        curve_layout.addWidget(self.curve_widget)
        curve_group.setLayout(curve_layout)
        main_layout.addWidget(curve_group, 2)

        # 第三行：实时数值显示 + API状态
        value_widget = QWidget()
        value_layout = QHBoxLayout()

        self.channel_value_labels = []
        channel_names = ['CH1 (右)', 'CH2 (上)', 'CH3 (左)', 'CH4 (中心)', 'CH5 (下)']
        colors = ['#ff5050', '#50c8ff', '#50ff50', '#ffc850', '#c850ff']
        for i in range(NUM_CHANNELS):
            label = QLabel(f'{channel_names[i]}: --')
            label.setFont(QFont('Courier', 11, QFont.Bold))
            label.setStyleSheet(f'color: {colors[i]}; padding: 5px; border: 1px solid #555; border-radius: 3px;')
            label.setMinimumWidth(150)
            value_layout.addWidget(label)
            self.channel_value_labels.append(label)

        # API状态标签
        self.api_status_label = QLabel('API: 未启动')
        self.api_status_label.setFont(QFont('Courier', 10, QFont.Bold))
        self.api_status_label.setStyleSheet('color: #888; padding: 5px; border: 1px solid #555; border-radius: 3px;')
        self.api_status_label.setMinimumWidth(180)
        value_layout.addWidget(self.api_status_label)

        value_widget.setLayout(value_layout)
        main_layout.addWidget(value_widget)

        panel.setLayout(main_layout)
        return panel

    def get_available_ports(self) -> list:
        """获取可用串口"""
        try:
            import serial.tools.list_ports
            ports = [p.device for p in serial.tools.list_ports.comports()]
            return ports if ports else ['COM1']
        except:
            return ['COM1', 'COM2', '/dev/ttyUSB0']

    def update_port_list(self):
        """更新可用串口列表"""
        current_port = self.port_combo.currentText()
        available_ports = self.get_available_ports()
        current_ports = [self.port_combo.itemText(i) for i in range(self.port_combo.count())]

        if set(current_ports) == set(available_ports):
            return

        new_ports = [p for p in available_ports if p not in current_ports]
        self.port_combo.clear()
        self.port_combo.addItems(available_ports)

        if new_ports:
            if not (self.communicator and self.communicator.is_connected):
                self.port_combo.setCurrentText(new_ports[0])
        else:
            if current_port in available_ports:
                self.port_combo.setCurrentText(current_port)

        if self.communicator and self.communicator.is_connected:
            connected_port = self.communicator.port
            if connected_port not in available_ports:
                self.on_disconnect()

    def on_connect(self):
        """连接/断开串口"""
        if self.communicator and self.communicator.is_connected:
            self.on_disconnect()
        else:
            self.on_serial_connect()

    def on_serial_connect(self):
        """建立串口连接"""
        port = self.port_combo.currentText()
        baudrate = int(self.baudrate_combo.currentText())
        timeout = self.timeout_spin.value()

        self.communicator = SerialCommunicator(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            channels=NUM_CHANNELS
        )

        if self.communicator.connect():
            self.connection_time = time.time()
            self.is_collecting_initial = True
            self.initial_data_collector = {ch: [] for ch in range(NUM_CHANNELS)}
            self.initialization_done = False

            self.receiver_thread = DataReceiverThread(self.communicator)
            self.receiver_thread.data_received.connect(self.on_data_received)
            self.receiver_thread.start()

            self.status_label.setText('已连接')
            self.status_label.setStyleSheet('color: green')
            self.connect_btn.setText('断开连接')

            # 启动API服务器
            if self.api_enabled and self.api_server:
                self.api_server.start()

    def on_initialize_zero(self):
        """初始化归零"""
        if self.latest_raw_data:
            for ch in range(NUM_CHANNELS):
                self.initial_values[ch] = self.latest_raw_data.get(ch, 0)
        else:
            self.initial_values = {ch: 0.0 for ch in range(NUM_CHANNELS)}

    def on_toggle_flip(self):
        """切换变量翻转"""
        for ch in self.channel_flip_states:
            self.channel_flip_states[ch] = not self.channel_flip_states[ch]

    def on_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于",
            "5通道圆形电容感应数据可视化系统\n\n"
            "版本: v1.6\n\n"
            "通道布局:\n"
            "  CH1 - 右侧扇形\n"
            "  CH2 - 上方扇形\n"
            "  CH3 - 左侧扇形\n"
            "  CH5 - 下方扇形\n"
            "  CH4 - 中心圆\n\n"
            "Copyright © 2026 宁波风隐科技有限公司")

    def on_disconnect(self):
        """断开串口连接"""
        if self.receiver_thread:
            self.receiver_thread.stop()
            self.receiver_thread.wait()
        if self.communicator:
            self.communicator.disconnect()

        # 停止API服务器
        if self.api_server:
            self.api_server.stop()

        self.status_label.setText('未连接')
        self.status_label.setStyleSheet('color: red')
        self.connect_btn.setText('连接串口')

    def check_heartbeat(self):
        """心跳检测"""
        if self.communicator and self.communicator.is_connected:
            current_time = time.time()
            elapsed = current_time - self.last_data_time
            if elapsed > self.heartbeat_timeout and self.initialization_done:
                print(f"[心跳检测] 信号断开，超时 {elapsed:.2f} 秒")
                self.on_disconnect()
                self.update_port_list()

    def calculate_force(self, channel_data: dict) -> tuple:
        """
        基于5通道电容值计算压力和剪切力 - 夹爪应用优化版
        
        传感器参数：
        - 直径17mm圆形电容传感器
        - 4层PCB + 2层FPC + 0.1mm膜材料
        - 应用场景：平板压力夹爪
        
        夹爪场景特点：
        - 只关心正向按压力（无负值）
        - 压力分布通常较均匀（平板夹爪）
        - 剪切力由夹爪夹持偏心产生
        
        物理模型：
        - 电容变化量 ΔC 与局部压力近似线性: ΔC ∝ ε₀·εr·A/d² · Δd
        - 总法向力 Fz = Σ(ΔCi × ki)
        - 剪切力通过质心偏移计算
        """
        # 传感器物理参数
        R_sensor = 8.5  # 传感器有效半径(mm)
        
        # 电极面积权重
        w_sector = 0.218
        w_center = 0.125
        
        # 获取各通道值（已取绝对值，只含正值或0）
        ch1 = channel_data.get(0, 0)  # 右侧扇形
        ch2 = channel_data.get(1, 0)  # 上方扇形
        ch3 = channel_data.get(2, 0)  # 左侧扇形
        ch4 = channel_data.get(3, 0)  # 中心圆
        ch5 = channel_data.get(4, 0)  # 下方扇形
        
        # 加权总压力
        raw_pressure = (ch1 * w_sector + ch2 * w_sector + ch3 * w_sector +
                        ch5 * w_sector + ch4 * w_center)
        
        # 线性补偿：夹爪平板接触，电容-力关系在小范围近似线性
        # 但在大压力下膜材料(0.1mm)压缩导致非线性
        # 使用二次多项式补偿提升线性度
        if raw_pressure > 0:
            # 补偿公式: P_comp = a*P_raw + b*P_raw^2
            # a=4.0 (线性增益), b=-0.0001 (非线性修正，大压力时略微降低增益)
            total_pressure = 4.0 * raw_pressure - 0.0001 * raw_pressure * raw_pressure
            total_pressure = max(total_pressure, 0)  # 确保非负
        else:
            total_pressure = 0
        
        if total_pressure < 5:
            return 0, 0, 0
        
        # 各电极几何位置
        r_sector = 0.675
        
        positions = {
            'ch1': (r_sector, 0),           # 右
            'ch2': (0, r_sector),           # 上
            'ch3': (-r_sector, 0),          # 左
            'ch5': (0, -r_sector),          # 下
            'ch4': (0, 0),                  # 中心
        }
        
        # 加权质心
        sum_wx = (ch1 * w_sector * positions['ch1'][0] +
                  ch2 * w_sector * positions['ch2'][0] +
                  ch3 * w_sector * positions['ch3'][0] +
                  ch5 * w_sector * positions['ch5'][0] +
                  ch4 * w_center * positions['ch4'][0])
        
        sum_wy = (ch1 * w_sector * positions['ch1'][1] +
                  ch2 * w_sector * positions['ch2'][1] +
                  ch3 * w_sector * positions['ch3'][1] +
                  ch5 * w_sector * positions['ch5'][1] +
                  ch4 * w_center * positions['ch4'][1])
        
        sum_w = (ch1 * w_sector + ch2 * w_sector + ch3 * w_sector +
                 ch5 * w_sector + ch4 * w_center)
        
        if sum_w < 1:
            return total_pressure, 0, 0
        
        cx = sum_wx / sum_w
        cy = sum_wy / sum_w
        
        # 剪切力（保留正负值，表示方向）
        shear_x = total_pressure * cx / r_sector
        shear_y = total_pressure * cy / r_sector
        
        # 非线性补偿
        compression_factor = 1.0 / (1.0 + 0.00005 * total_pressure)
        shear_x *= compression_factor
        shear_y *= compression_factor
        
        return total_pressure, shear_x, shear_y

    def on_data_received(self, data: dict):
        """处理接收到的数据"""
        self.packet_count += 1
        self.last_data_time = time.time()

        channel_data = data.get('channels', {})

        if not channel_data:
            channel_data = {i: np.random.randint(0, 1000) for i in range(NUM_CHANNELS)}

        # 保存原始数据
        self.latest_raw_data = channel_data.copy()

        # 应用滤波
        filtered_channel_data = {}
        for ch, value in channel_data.items():
            filter_type = self.filter_combo.currentText()
            if filter_type != '无' and ch in self.streaming_filters:
                self.streaming_filters[ch].algorithm = filter_type
                self.streaming_filters[ch].window_size = self.window_spin.value()
                filtered_value = self.streaming_filters[ch].filter(value)
                filtered_channel_data[ch] = filtered_value
            else:
                filtered_channel_data[ch] = value

        channel_data = filtered_channel_data

        # 初始数据收集
        current_time = time.time()
        if self.is_collecting_initial and self.connection_time:
            elapsed_time = current_time - self.connection_time
            for ch, value in channel_data.items():
                if ch in self.initial_data_collector:
                    self.initial_data_collector[ch].append(value)

            if elapsed_time >= 0.1:
                for ch in range(NUM_CHANNELS):
                    if self.initial_data_collector[ch]:
                        self.initial_values[ch] = np.mean(self.initial_data_collector[ch])
                    else:
                        self.initial_values[ch] = 0.0
                self.is_collecting_initial = False
                self.initialization_done = True
                self.last_data_time = time.time()

        # 归零处理 + 取绝对值（夹爪场景只关心按压力大小，不关心方向）
        processed_channel_data = {}
        for ch, value in channel_data.items():
            zeroed_value = value - self.initial_values[ch] if self.use_initial_zero else value
            if self.channel_flip_states.get(ch, False):
                zeroed_value = -zeroed_value
            # 静默平抑：±5以内的跳动视为噪声，平抑为零
            if abs(zeroed_value) <= 5.0:
                zeroed_value = 0.0
            # 夹爪应用：只取正值或0，负值视为0（无压力）
            processed_channel_data[ch] = max(zeroed_value, 0.0)

        # 迟滞补偿：检测释放趋势，加速回归零位
        # 当所有通道值都在下降且接近零时，加速衰减
        if hasattr(self, '_prev_processed') and self._prev_processed:
            all_declining = True
            for ch in range(NUM_CHANNELS):
                curr = processed_channel_data.get(ch, 0)
                prev = self._prev_processed.get(ch, 0)
                if curr > prev + 2:  # 如果有通道在上升，则不在释放趋势中
                    all_declining = False
                    break
            if all_declining:
                max_val = max(processed_channel_data.values())
                if max_val < 50:  # 接近零位的小波动
                    # 加速衰减：将小值快速拉向0
                    decay_factor = 0.7  # 衰减系数
                    for ch in range(NUM_CHANNELS):
                        processed_channel_data[ch] *= decay_factor
                elif max_val < 150:  # 中等值，轻度衰减
                    decay_factor = 0.9
                    for ch in range(NUM_CHANNELS):
                        processed_channel_data[ch] *= decay_factor
        self._prev_processed = processed_channel_data.copy()

        # 计算压力和剪切力
        total_pressure, shear_x, shear_y = self.calculate_force(processed_channel_data)

        # 无压力自动归零机制
        # 条件：所有通道值 < 100 且持续 0.1秒
        all_below_threshold = all(processed_channel_data.get(ch, 0) < 100.0
                                  for ch in range(NUM_CHANNELS))

        if all_below_threshold:
            if self.no_pressure_start_time is None:
                self.no_pressure_start_time = current_time
            elif (current_time - self.no_pressure_start_time) >= 0.1:
                # 0.1秒超时，立即重新归零
                for ch in range(NUM_CHANNELS):
                    self.initial_values[ch] = channel_data.get(ch, 0)
                self.no_pressure_start_time = None
                # 归零后清空显示
                processed_channel_data = {ch: 0.0 for ch in range(NUM_CHANNELS)}
                total_pressure, shear_x, shear_y = 0, 0, 0
        else:
            # 有压力信号，重置计时器
            self.no_pressure_start_time = None

        # 微小跳动过滤：低于30的噪声不影响除曲线图外的图形显示
        # 曲线图使用原始processed数据（上面已更新），其他图形使用过滤后数据
        has_real_force = any(processed_channel_data.get(ch, 0) >= 30.0 for ch in range(NUM_CHANNELS))
        if not has_real_force:
            # 所有通道都在30以下（微小跳动），其他图形显示全零
            display_channel_data = {ch: 0.0 for ch in range(NUM_CHANNELS)}
            display_pressure, display_shear_x, display_shear_y = 0, 0, 0
        else:
            # 有有效信号，将低于30的通道值置零（消除噪声影响），保留有效通道
            display_channel_data = {}
            for ch in range(NUM_CHANNELS):
                display_channel_data[ch] = processed_channel_data[ch] if processed_channel_data[ch] >= 30.0 else 0.0
            display_pressure, display_shear_x, display_shear_y = total_pressure, shear_x, shear_y

        # 更新3D受力场（内含迷你热力图）- 使用过滤后数据
        self.force_field_widget.update_data(display_channel_data, display_shear_x, display_shear_y, display_pressure)

        # 更新柱状图 - 使用过滤后数据
        self.bar_chart.update_data(display_pressure, display_shear_x, display_shear_y)

        # 更新曲线图 - 使用原始数据（含微小跳动，保持完整波形）
        for ch, value in processed_channel_data.items():
            self.curve.update_channel(ch, value, current_time)

        # 更新实时数值显示
        channel_names = ['CH1', 'CH2', 'CH3', 'CH4', 'CH5']
        for i in range(NUM_CHANNELS):
            val = processed_channel_data.get(i, 0)
            self.channel_value_labels[i].setText(f'{channel_names[i]}: {val:.1f}')

        # 存储到数据库
        timestamp = data.get('timestamp', datetime.now().isoformat())
        self.db_manager.add_record(timestamp, channel_data)

        # 更新API数据（原始数据 + 力学计算数据）
        if self.api_enabled and self.api_server:
            # 发送原始传感器数据（使用归零前的原始数据，更真实）
            self.api_server.update_raw_data(channel_data)
            # 发送力学计算数据
            # 从3D受力场控件获取质心和半径信息
            fc_x = getattr(self.force_field_widget, 'force_center_x', 0.0)
            fc_y = getattr(self.force_field_widget, 'force_center_y', 0.0)
            fc_r = getattr(self.force_field_widget, 'force_radius', 0.0)
            self.api_server.update_force_data(
                total_pressure=total_pressure,
                shear_x=shear_x,
                shear_y=shear_y,
                center_x=fc_x,
                center_y=fc_y,
                radius=fc_r
            )

    def on_save_data(self):
        """保存数据"""
        QMessageBox.information(self, '已保存', f'已缓存 {self.db_manager.get_row_count()} 条记录')

    def on_export_csv(self):
        """导出CSV"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, '导出数据', '', 'CSV Files (*.csv)'
        )
        if file_path:
            self.db_manager.export_to_csv(file_path)
            QMessageBox.information(self, '成功', f'已导出到 {file_path}')

    def on_clear_data(self):
        """清空数据"""
        reply = QMessageBox.question(
            self, '确认', '确定要清空所有数据吗？',
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.db_manager.flush()
            self.packet_count = 0
            self.curve.clear()
            self.force_field_widget.update_data({ch: 0.0 for ch in range(NUM_CHANNELS)}, 0, 0, 0)
            self.bar_chart.update_data(0, 0, 0)
            QMessageBox.information(self, '已清空', '数据已清空')

    def update_display(self):
        """更新显示"""
        # 更新API状态标签
        if self.api_server and self.api_server.is_alive():
            self.api_status_label.setText(f'API: 运行中 :{self.api_server.port}')
            self.api_status_label.setStyleSheet('color: #0f0; padding: 5px; border: 1px solid #555; border-radius: 3px;')
        else:
            self.api_status_label.setText('API: 未启动')
            self.api_status_label.setStyleSheet('color: #888; padding: 5px; border: 1px solid #555; border-radius: 3px;')

        if self.communicator and self.communicator.is_connected:
            if self.latest_raw_data:
                stats_lines = "系统状态\n" + "─" * 35 + "\n"
                stats_lines += f"通信状态: 连接中\n"
                stats_lines += f"接收数据包: {self.packet_count}\n\n"
                stats_lines += "通道实时值\n" + "─" * 35 + "\n"
                channel_names = ['CH1(右)', 'CH2(上)', 'CH3(左)', 'CH4(中心)', 'CH5(下)']
                for ch in range(NUM_CHANNELS):
                    raw = self.latest_raw_data.get(ch, 0)
                    init = self.initial_values.get(ch, 0)
                    delta = raw - init
                    if self.channel_flip_states.get(ch, False):
                        delta = -delta
                    # 静默平抑：±5以内视为噪声
                    if abs(delta) <= 5.0:
                        delta = 0.0
                    stats_lines += f"{channel_names[ch]}: {delta:>8.1f} (原始: {raw:.0f})\n"
                self.stats_text.setText(stats_lines)
        else:
            stats_text = f"""
系统统计
{'─' * 35}
接收数据包: {self.packet_count}
错误数: {self.error_count}
数据库记录: {self.db_manager.get_row_count()}

通信状态
{'─' * 35}
未连接
            """
            self.stats_text.setText(stats_text)

    def on_mirror_flip(self):
        """3D受力场镜像翻转"""
        self.force_mirror_h = not self.force_mirror_h
        self.force_mirror_v = not self.force_mirror_v
        # 通知3D受力场控件更新翻转状态
        self.force_field_widget.set_mirror(self.force_mirror_h, self.force_mirror_v)
        # 更新显示
        if self.latest_raw_data:
            # 触发一次数据更新以应用翻转
            self.on_data_received({'channels': self.latest_raw_data})

    def on_x_window_changed(self, value):
        """修改曲线图X轴时间窗口"""
        self.curve.x_window = float(value)

    def show_curve_context_menu(self, position):
        """曲线图右键菜单"""
        from PyQt5.QtWidgets import QMenu
        menu = QMenu()
        auto_range_action = menu.addAction("自动缩放")
        reset_view_action = menu.addAction("重置视图")
        menu.addSeparator()
        export_image_action = menu.addAction("导出图像")
        toggle_legend_action = menu.addAction("显示/隐藏图例")

        auto_range_action.triggered.connect(lambda: self.curve_plot.autoRange())
        reset_view_action.triggered.connect(lambda: self.curve_plot.setRange(xRange=None, yRange=None, padding=0.1))

        def export_image():
            file_path, _ = QFileDialog.getSaveFileName(
                self, "导出图像", "", "PNG Files (*.png);;JPEG Files (*.jpg)"
            )
            if file_path:
                self.curve_plot.save(file_path)

        export_image_action.triggered.connect(export_image)

        def toggle_legend():
            legend = self.curve_plot.legend
            if legend:
                legend.setVisible(not legend.isVisible())

        toggle_legend_action.triggered.connect(toggle_legend)
        menu.exec_(self.curve_widget.mapToGlobal(position))

    def closeEvent(self, event):
        """关闭窗口"""
        self.on_disconnect()
        self.db_manager.close()
        event.accept()


def main():
    app = QApplication(sys.argv)

    # 单实例锁
    try:
        import win32event
        import win32api
        from winerror import ERROR_ALREADY_EXISTS
        mutex_name = "Global\WINSTEL_KF_5P_CircularSensor"
        mutex = win32event.CreateMutex(None, True, mutex_name)
        if win32api.GetLastError() == ERROR_ALREADY_EXISTS:
            QMessageBox.critical(None, "错误", "程序已在运行中！请关闭现有实例后再启动。")
            sys.exit(1)
    except ImportError:
        print("警告：未安装pywin32，无法启用单实例保护")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
