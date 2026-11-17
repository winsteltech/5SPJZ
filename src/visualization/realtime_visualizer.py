#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时数据可视化模块 - 5通道圆形电容感应
3D受力场可视化 + 迷你热力图 + 双向柱状图 + 曲线图
"""

import numpy as np
import math
import random
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    import pyqtgraph as pg
    from PyQt5 import QtCore, QtGui, QtWidgets
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    logger.warning("PyQtGraph 未安装，可视化功能不可用")

# 通道颜色
CHANNEL_COLORS = [
    (255, 80, 80),    # 红色 - CH1
    (80, 200, 255),   # 蓝色 - CH2
    (80, 255, 80),    # 绿色 - CH3
    (255, 200, 80),   # 黄色 - CH4
    (200, 80, 255),   # 紫色 - CH5
]

# 通道几何位置（归一化坐标，基于17mm直径传感器）
# CH1=右, CH2=上, CH3=左, CH4=中心, CH5=下
CHANNEL_POSITIONS = {
    0: (0.675, 0.0),     # CH1 右侧扇形
    1: (0.0, 0.675),     # CH2 上方扇形
    2: (-0.675, 0.0),    # CH3 左侧扇形
    3: (0.0, 0.0),       # CH4 中心圆
    4: (0.0, -0.675),    # CH5 下方扇形
}

# 电极面积权重
CHANNEL_WEIGHTS = {
    0: 0.218,  # 扇形
    1: 0.218,
    2: 0.218,
    3: 0.125,  # 中心圆
    4: 0.218,
}

# 微小跳动阈值 - 低于此值的信号视为噪声
NOISE_THRESHOLD = 30.0


class MiniHeatmapWidget(QtWidgets.QWidget):
    """
    迷你3D热力图控件 - 缩小版放在左下角
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.channels = 5
        self.data = [0.0] * 5
        self.shear_x = 0.0
        self.shear_y = 0.0
        self.total_pressure = 0.0

        self.color_min = 0.0
        self.color_max = 1500.0
        self.gap_deg = 4.0

        half_gap = self.gap_deg / 2.0
        self.sector_angles = [
            (-45.0 + half_gap, 45.0 - half_gap),
            (45.0 + half_gap, 135.0 - half_gap),
            (135.0 + half_gap, 225.0 - half_gap),
            (225.0 + half_gap, 315.0 - half_gap),
        ]

        self.setMinimumSize(140, 140)
        self.setMaximumSize(180, 180)

    def update_data(self, channel_data: dict, shear_x: float = 0.0, shear_y: float = 0.0, total_pressure: float = 0.0):
        for i in range(self.channels):
            self.data[i] = float(channel_data.get(i, 0.0))
        self.shear_x = shear_x
        self.shear_y = shear_y
        self.total_pressure = total_pressure
        self.update()

    def _value_to_color(self, value):
        value = max(value, 0.0)
        if self.color_max == self.color_min:
            return QtGui.QColor(30, 30, 30)
        norm = np.clip((value - self.color_min) / (self.color_max - self.color_min), 0, 1)

        if norm < 0.2:
            t = norm / 0.2
            return QtGui.QColor(0, 0, int(50 + t * 150))
        elif norm < 0.4:
            t = (norm - 0.2) / 0.2
            return QtGui.QColor(0, int(t * 200), 255)
        elif norm < 0.6:
            t = (norm - 0.4) / 0.2
            return QtGui.QColor(int(t * 255), 255, int(255 - t * 128))
        elif norm < 0.8:
            t = (norm - 0.6) / 0.2
            return QtGui.QColor(255, int(255 - t * 128), 0)
        else:
            t = (norm - 0.8) / 0.2
            return QtGui.QColor(255, int(127 + t * 128), int(t * 128))

    def _draw_sector(self, painter, cx, cy, inner_r, outer_r, start_deg, end_deg, value):
        path = QtGui.QPainterPath()
        start_rad = math.radians(start_deg)
        end_rad = math.radians(end_deg)

        ox1 = cx + outer_r * math.cos(start_rad)
        oy1 = cy - outer_r * math.sin(start_rad)
        ox2 = cx + outer_r * math.cos(end_rad)
        oy2 = cy - outer_r * math.sin(end_rad)
        ix1 = cx + inner_r * math.cos(end_rad)
        iy1 = cy - inner_r * math.sin(end_rad)
        ix2 = cx + inner_r * math.cos(start_rad)
        iy2 = cy - inner_r * math.sin(start_rad)

        path.moveTo(ox1, oy1)
        path.arcTo(cx - outer_r, cy - outer_r, outer_r * 2, outer_r * 2, start_deg, end_deg - start_deg)
        path.lineTo(ix1, iy1)
        path.arcTo(cx - inner_r, cy - inner_r, inner_r * 2, inner_r * 2, end_deg, start_deg - end_deg)
        path.closeSubpath()

        color = self._value_to_color(value)
        painter.setBrush(QtGui.QBrush(color))
        painter.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60), 1))
        painter.drawPath(path)

    def _draw_circle(self, painter, cx, cy, radius, value):
        color = self._value_to_color(value)
        painter.setBrush(QtGui.QBrush(color))
        painter.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60), 1))
        painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

    def _draw_force_arrow(self, painter, cx, cy, outer_r):
        if abs(self.shear_x) < 5 and abs(self.shear_y) < 5:
            return

        max_shear = max(abs(self.shear_x), abs(self.shear_y), 1)
        arrow_scale = min(outer_r * 0.5 / max_shear, 2.0)

        start_x, start_y = cx, cy
        end_x = cx + self.shear_x * arrow_scale
        end_y = cy - self.shear_y * arrow_scale

        pen = QtGui.QPen(QtGui.QColor(255, 255, 0), 2)
        painter.setPen(pen)
        painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))

        angle = math.atan2(start_y - end_y, end_x - start_x)
        arrow_len = 6
        arrow_angle = 0.5

        x1 = end_x - arrow_len * math.cos(angle - arrow_angle)
        y1 = end_y + arrow_len * math.sin(angle - arrow_angle)
        x2 = end_x - arrow_len * math.cos(angle + arrow_angle)
        y2 = end_y + arrow_len * math.sin(angle + arrow_angle)

        arrow_path = QtGui.QPainterPath()
        arrow_path.moveTo(end_x, end_y)
        arrow_path.lineTo(x1, y1)
        arrow_path.lineTo(x2, y2)
        arrow_path.closeSubpath()

        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 0)))
        painter.drawPath(arrow_path)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        painter.fillRect(self.rect(), QtGui.QColor(15, 15, 20))

        cx = self.width() / 2.0
        cy = self.height() / 2.0
        outer_r = min(cx, cy) * 0.75
        inner_r = min(cx, cy) * 0.28

        sector_data_indices = [0, 1, 2, 4]
        for i, (start_deg, end_deg) in enumerate(self.sector_angles):
            self._draw_sector(painter, cx, cy, inner_r + 2, outer_r,
                            start_deg, end_deg, self.data[sector_data_indices[i]])

        self._draw_circle(painter, cx, cy, inner_r, self.data[3])
        self._draw_force_arrow(painter, cx, cy, outer_r)


class ForceField3DWidget(QtWidgets.QWidget):
    """
    3D受力场可视化控件 - 高性能版本
    
    通过加权质心算法计算受力区域中心，
    用简洁的椭圆+渐变表现受力区域和大小，
    用箭头表现剪切方向和剪切力大小。
    
    左下角包含迷你热力图。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.channels = 5
        self.data = [0.0] * 5
        self.shear_x = 0.0
        self.shear_y = 0.0
        self.total_pressure = 0.0

        # 受力区域计算结果
        self.force_center_x = 0.0
        self.force_center_y = 0.0
        self.force_radius = 0.0
        self.force_magnitude = 0.0

        # 镜像翻转状态
        self.mirror_h = False  # 水平翻转（左右）
        self.mirror_v = False  # 垂直翻转（上下）

        # 预计算颜色映射表（避免每帧重复计算）
        self._color_cache = {}

        # 创建迷你热力图（左下角浮动）
        self.mini_heatmap = MiniHeatmapWidget(self)
        self.mini_heatmap.setParent(self)
        self.mini_heatmap.setGeometry(8, self.height() - 155, 145, 145)
        self.mini_heatmap.raise_()

        self.setMinimumSize(400, 350)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # 动画定时器 - 降低频率到150ms（约6.7fps动画，足够流畅）
        self.anim_tick = 0
        self.anim_timer = QtCore.QTimer(self)
        self.anim_timer.timeout.connect(self._anim_update)
        self.anim_timer.start(150)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.mini_heatmap.setGeometry(8, self.height() - 155, 145, 145)

    def _anim_update(self):
        self.anim_tick += 1
        self.update()

    def set_mirror(self, mirror_h: bool, mirror_v: bool):
        """设置镜像翻转状态"""
        self.mirror_h = mirror_h
        self.mirror_v = mirror_v
        self.update()

    def update_data(self, channel_data: dict, shear_x: float = 0.0, shear_y: float = 0.0, total_pressure: float = 0.0):
        for i in range(self.channels):
            self.data[i] = float(channel_data.get(i, 0.0))
        self.shear_x = shear_x
        self.shear_y = shear_y
        self.total_pressure = total_pressure

        # 计算受力区域
        self._compute_force_field()

        # 同步更新迷你热力图
        self.mini_heatmap.update_data(channel_data, shear_x, shear_y, total_pressure)
        self.update()

    def _compute_force_field(self):
        """计算受力区域特征（简化版）"""
        sum_w = 0.0
        sum_wx = 0.0
        sum_wy = 0.0
        sum_wx2 = 0.0
        sum_wy2 = 0.0
        peak_val = 0.0

        for ch in range(self.channels):
            val = self.data[ch]
            w = CHANNEL_WEIGHTS.get(ch, 0.2)
            px, py = CHANNEL_POSITIONS.get(ch, (0, 0))

            sum_w += val * w
            sum_wx += val * w * px
            sum_wy += val * w * py
            sum_wx2 += val * w * px * px
            sum_wy2 += val * w * py * py
            peak_val = max(peak_val, val)

        if sum_w < 1.0:
            self.force_center_x = 0.0
            self.force_center_y = 0.0
            self.force_radius = 0.0
            self.force_magnitude = 0.0
            return

        self.force_center_x = sum_wx / sum_w
        self.force_center_y = sum_wy / sum_w

        var_x = sum_wx2 / sum_w - self.force_center_x ** 2
        var_y = sum_wy2 / sum_w - self.force_center_y ** 2
        self.force_radius = math.sqrt(max(var_x, 0.0) + max(var_y, 0.0))

        max_possible = 2000.0
        self.force_magnitude = min(peak_val / max_possible, 1.0)

    def _fast_pressure_color(self, norm):
        """快速颜色映射（简化版，减少分支）"""
        norm = max(0.0, min(norm, 1.0))
        # 简化：深蓝->青->绿->黄->红
        if norm < 0.33:
            t = norm / 0.33
            return QtGui.QColor(0, int(t * 200), int(50 + t * 205))
        elif norm < 0.66:
            t = (norm - 0.33) / 0.33
            return QtGui.QColor(int(t * 255), 200, int(255 - t * 200))
        else:
            t = (norm - 0.66) / 0.34
            return QtGui.QColor(255, int(200 - t * 150), int(t * 50))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        cx = w / 2.0
        cy = h / 2.0
        radius = min(cx, cy) * 0.72

        # 深色背景
        painter.fillRect(self.rect(), QtGui.QColor(10, 12, 18))

        # === 传感器圆形轮廓 ===
        pen = QtGui.QPen(QtGui.QColor(0, 100, 150, 80), 1.5)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

        # 内圈
        inner_r = radius * 0.38
        pen = QtGui.QPen(QtGui.QColor(0, 100, 150, 50), 1)
        painter.setPen(pen)
        painter.drawEllipse(QtCore.QPointF(cx, cy), inner_r, inner_r)

        # 十字参考线
        pen = QtGui.QPen(QtGui.QColor(0, 80, 120, 40), 1, QtCore.Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(int(cx - radius), int(cy), int(cx + radius), int(cy))
        painter.drawLine(int(cx), int(cy - radius), int(cx), int(cy + radius))

        # 同心圆参考线（减少为2条）
        for i in [2, 4]:
            r = radius * i / 4.0
            pen = QtGui.QPen(QtGui.QColor(0, 80, 120, 25), 1, QtCore.Qt.DotLine)
            painter.setPen(pen)
            painter.drawEllipse(QtCore.QPointF(cx, cy), r, r)

        # === 简化版受力区域表现 ===
        # 计算峰值和归一化
        peak_val = max(self.data) if any(self.data) else 0

        if peak_val > NOISE_THRESHOLD:
            # 受力中心位置（应用镜像翻转）
            fcx = cx + (self.force_center_x * (-1 if self.mirror_h else 1)) * radius
            fcy = cy - (self.force_center_y * (-1 if self.mirror_v else 1)) * radius

            # 等效半径（像素）
            bump_rx = max(self.force_radius * radius * 1.8, radius * 0.12)
            bump_ry = bump_rx

            # 脉冲动画（简化）
            pulse = 1.0 + 0.03 * math.sin(self.anim_tick * 0.12)

            # === 1. 压力区域填充（3层同心椭圆，代替等高线） ===
            levels = [
                (0.25, 0.20),
                (0.50, 0.35),
                (0.80, 0.55),
            ]

            for level, alpha_scale in levels:
                rx = bump_rx * (1.0 + (1.0 - level) * 0.5) * pulse
                ry = bump_ry * (1.0 + (1.0 - level) * 0.5) * pulse
                color = self._fast_pressure_color(level)
                fill_color = QtGui.QColor(color.red(), color.green(), color.blue(),
                                          int(80 * alpha_scale * self.force_magnitude))
                painter.setBrush(QtGui.QBrush(fill_color))
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawEllipse(QtCore.QPointF(fcx, fcy), rx, ry)

            # === 2. 3D凸起主体（简化：单层渐变椭圆+高光） ===
            bump_gradient = QtGui.QRadialGradient(
                fcx - bump_rx * 0.15, fcy - bump_ry * 0.15,
                bump_rx * 1.1
            )
            peak_color = self._fast_pressure_color(1.0)
            bump_gradient.setColorAt(0, QtGui.QColor(
                min(255, peak_color.red() + 60),
                min(255, peak_color.green() + 60),
                min(255, peak_color.blue() + 60),
                int(160 * self.force_magnitude)
            ))
            bump_gradient.setColorAt(1.0, QtGui.QColor(
                peak_color.red() // 2, peak_color.green() // 2, peak_color.blue() // 2, 0
            ))

            painter.setBrush(QtGui.QBrush(bump_gradient))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(QtCore.QPointF(fcx, fcy), bump_rx * pulse, bump_ry * pulse)

            # 高光点
            highlight = QtGui.QRadialGradient(
                fcx - bump_rx * 0.2, fcy - bump_ry * 0.2,
                bump_rx * 0.25
            )
            highlight.setColorAt(0, QtGui.QColor(255, 255, 255, int(80 * self.force_magnitude)))
            highlight.setColorAt(1, QtGui.QColor(255, 255, 255, 0))
            painter.setBrush(QtGui.QBrush(highlight))
            painter.drawEllipse(QtCore.QPointF(fcx - bump_rx * 0.1, fcy - bump_ry * 0.1),
                              bump_rx * 0.3, bump_ry * 0.25)

            # === 3. 受力中心标记 ===
            center_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 180), 1.5)
            painter.setPen(center_pen)
            cross_size = 6
            painter.drawLine(int(fcx - cross_size), int(fcy), int(fcx + cross_size), int(fcy))
            painter.drawLine(int(fcx), int(fcy - cross_size), int(fcx), int(fcy + cross_size))

            # === 4. 剪切力箭头（应用镜像翻转） ===
            if abs(self.shear_x) > 1 or abs(self.shear_y) > 1:
                shear_mag = math.sqrt(self.shear_x ** 2 + self.shear_y ** 2)
                max_arrow_len = radius * 0.55
                arrow_len = min(shear_mag / max(shear_mag, 1) * max_arrow_len, max_arrow_len)

                # 应用镜像翻转到剪切力方向
                dir_x = (self.shear_x * (-1 if self.mirror_h else 1)) / max(shear_mag, 1)
                dir_y = (self.shear_y * (-1 if self.mirror_v else 1)) / max(shear_mag, 1)

                arrow_end_x = fcx + dir_x * arrow_len
                arrow_end_y = fcy - dir_y * arrow_len

                # 箭头线（纯色，减少渐变开销）
                arrow_pen = QtGui.QPen(QtGui.QColor(255, 180, 50, 230), 2.5)
                painter.setPen(arrow_pen)
                painter.drawLine(int(fcx), int(fcy), int(arrow_end_x), int(arrow_end_y))

                # 箭头头部
                angle = math.atan2(fcy - arrow_end_y, arrow_end_x - fcx)
                head_len = 8
                head_angle = 0.45

                hx1 = arrow_end_x - head_len * math.cos(angle - head_angle)
                hy1 = arrow_end_y + head_len * math.sin(angle - head_angle)
                hx2 = arrow_end_x - head_len * math.cos(angle + head_angle)
                hy2 = arrow_end_y + head_len * math.sin(angle + head_angle)

                arrow_head = QtGui.QPainterPath()
                arrow_head.moveTo(arrow_end_x, arrow_end_y)
                arrow_head.lineTo(hx1, hy1)
                arrow_head.lineTo(hx2, hy2)
                arrow_head.closeSubpath()
                painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 180, 50, 230)))
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawPath(arrow_head)

                # 剪切力数值
                font = QtGui.QFont("Arial", 8, QtGui.QFont.Bold)
                painter.setFont(font)
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 200, 80)))
                painter.drawText(int(arrow_end_x + 4), int(arrow_end_y - 4), f"F:{shear_mag:.0f}")

        # === 通道数据点（5个电极位置，应用镜像翻转） ===
        for ch in range(self.channels):
            px, py = CHANNEL_POSITIONS.get(ch, (0, 0))
            # 应用镜像翻转
            px = px * (-1 if self.mirror_h else 1)
            py = py * (-1 if self.mirror_v else 1)
            sx = cx + px * radius
            sy = cy - py * radius
            val = self.data[ch]

            # 电极点发光效果（简化）
            if val > NOISE_THRESHOLD:
                glow_r = 5 + min(val / 600, 6)
                c = CHANNEL_COLORS[ch]
                intensity = min(val / 1000, 1.0)
                glow = QtGui.QRadialGradient(sx, sy, glow_r)
                glow.setColorAt(0, QtGui.QColor(c[0], c[1], c[2], int(180 * intensity)))
                glow.setColorAt(1, QtGui.QColor(c[0], c[1], c[2], 0))
                painter.setBrush(QtGui.QBrush(glow))
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawEllipse(QtCore.QPointF(sx, sy), glow_r, glow_r)

            # 电极点核心
            painter.setBrush(QtGui.QBrush(QtGui.QColor(200, 200, 200, 180)))
            painter.setPen(QtGui.QPen(QtGui.QColor(CHANNEL_COLORS[ch][0], CHANNEL_COLORS[ch][1], CHANNEL_COLORS[ch][2]), 1.5))
            painter.drawEllipse(QtCore.QPointF(sx, sy), 3, 3)

        # === 信息文字 ===
        font = QtGui.QFont("Arial", 9, QtGui.QFont.Bold)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 200, 255)))
        painter.drawText(int(radius * 0.1), 18, "3D受力场分析")

        # 右上角信息面板
        info_x = cx + radius * 0.35
        info_y = 25
        font = QtGui.QFont("Consolas", 8)
        painter.setFont(font)

        info_lines = [
            (f"总压力: {self.total_pressure:.0f}", QtGui.QColor(0, 200, 255)),
            (f"剪切X: {self.shear_x:.0f}", QtGui.QColor(255, 100, 100)),
            (f"剪切Y: {self.shear_y:.0f}", QtGui.QColor(100, 255, 100)),
            (f"质心X: {self.force_center_x:.3f}", QtGui.QColor(200, 200, 200)),
            (f"质心Y: {self.force_center_y:.3f}", QtGui.QColor(200, 200, 200)),
            (f"等效半径: {self.force_radius:.3f}", QtGui.QColor(200, 200, 200)),
        ]

        info_bg_w = 130
        info_bg_h = len(info_lines) * 16 + 8
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 100)))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 100, 150, 80), 1))
        painter.drawRoundedRect(int(info_x - 5), int(info_y - 3), info_bg_w, info_bg_h, 4, 4)

        for i, (text, color) in enumerate(info_lines):
            painter.setPen(QtGui.QPen(color))
            painter.drawText(int(info_x), int(info_y + i * 16 + 12), text)

        # === 迷你热力图边框 ===
        hm_rect = self.mini_heatmap.geometry()
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 100, 150, 60), 1))
        painter.drawRoundedRect(hm_rect.x() - 2, hm_rect.y() - 2, hm_rect.width() + 4, hm_rect.height() + 4, 4, 4)

        # 迷你热力图标签
        font = QtGui.QFont("Arial", 7)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 150, 200, 150)))
        painter.drawText(hm_rect.x(), hm_rect.y() - 4, "传感器分布")


class BarChartWidget(QtWidgets.QWidget):
    """
    双向柱状图控件 - 修复总压力超出范围问题
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.total_pressure = 0.0
        self.shear_force_x = 0.0
        self.shear_force_y = 0.0
        self.shear_magnitude = 0.0

        self.use_auto_range = True
        self.fixed_max_range = 5000.0
        self.history_values = []
        self.max_history = 30

        self.setMinimumSize(220, 300)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def update_data(self, total_pressure: float, shear_x: float, shear_y: float):
        self.total_pressure = total_pressure
        self.shear_force_x = shear_x
        self.shear_force_y = shear_y
        self.shear_magnitude = math.sqrt(shear_x**2 + shear_y**2)

        self.history_values.append(max(abs(total_pressure), abs(shear_x), abs(shear_y), self.shear_magnitude))
        if len(self.history_values) > self.max_history:
            self.history_values.pop(0)

        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        painter.fillRect(self.rect(), QtGui.QColor(20, 20, 20))

        w = self.width()
        h = self.height()
        margin = 15
        chart_w = w - margin * 2
        chart_h = h - margin * 2 - 50

        # 标题
        font = QtGui.QFont("Arial", 11, QtGui.QFont.Bold)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        painter.drawText(margin, 5, chart_w, 25, QtCore.Qt.AlignCenter, "力学分析")

        # 确定量程
        if self.use_auto_range and self.history_values:
            max_val = max(max(self.history_values) * 1.3, 100.0)
        else:
            max_val = self.fixed_max_range
        max_val = min(max_val, 20000.0)

        center_y = margin + 30 + chart_h / 2

        # 中心参考线
        center_pen = QtGui.QPen(QtGui.QColor(100, 100, 100), 2, QtCore.Qt.DashLine)
        painter.setPen(center_pen)
        painter.drawLine(int(margin), int(center_y), int(margin + chart_w), int(center_y))

        # 网格线
        grid_pen = QtGui.QPen(QtGui.QColor(60, 60, 60), 1, QtCore.Qt.DotLine)
        painter.setPen(grid_pen)
        for i in range(-4, 5):
            if i == 0:
                continue
            y_line = center_y - (i / 4.0) * (chart_h / 2)
            if margin + 30 < y_line < margin + 30 + chart_h:
                painter.drawLine(int(margin), int(y_line), int(margin + chart_w), int(y_line))

        scale = (chart_h / 2) / max_val
        bar_width = chart_w / 4

        bars = [
            ("总压力", self.total_pressure, QtGui.QColor(0, 150, 255), "up"),
            ("剪切合", self.shear_magnitude, QtGui.QColor(255, 200, 50), "up"),
            ("剪切X", self.shear_force_x, QtGui.QColor(255, 100, 100), "bidir"),
            ("剪切Y", self.shear_force_y, QtGui.QColor(100, 255, 100), "bidir"),
        ]

        for i, (label, value, color, bar_type) in enumerate(bars):
            x = margin + i * bar_width + bar_width * 0.12
            bar_w = bar_width * 0.76

            display_value = value
            if bar_type == "up":
                bar_h = min(abs(display_value) * scale, chart_h)
                y = center_y - bar_h

                painter.setBrush(QtGui.QBrush(color))
                painter.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200), 1))
                painter.drawRect(int(x), int(y), int(bar_w), int(bar_h))

                font = QtGui.QFont("Arial", 8, QtGui.QFont.Bold)
                painter.setFont(font)
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
                val_text = f"{display_value:.0f}"
                label_y = int(y) - 14
                painter.drawText(int(x), label_y, int(bar_w), 14,
                               QtCore.Qt.AlignCenter, val_text)

            else:
                bar_h = min(abs(display_value) * scale, chart_h / 2)

                if display_value >= 0:
                    y = center_y - bar_h
                    painter.setBrush(QtGui.QBrush(color))
                else:
                    y = center_y
                    dark_color = QtGui.QColor(
                        max(0, color.red() - 80),
                        max(0, color.green() - 80),
                        max(0, color.blue() - 80)
                    )
                    painter.setBrush(QtGui.QBrush(dark_color))

                painter.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200), 1))
                painter.drawRect(int(x), int(y), int(bar_w), int(bar_h))

                font = QtGui.QFont("Arial", 8, QtGui.QFont.Bold)
                painter.setFont(font)
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
                val_text = f"{display_value:.0f}"
                if display_value >= 0:
                    label_y = int(y) - 14
                else:
                    label_y = int(y) + int(bar_h) + 2
                painter.drawText(int(x), label_y, int(bar_w), 14,
                               QtCore.Qt.AlignCenter, val_text)

            # 底部标签
            font = QtGui.QFont("Arial", 8)
            painter.setFont(font)
            painter.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200)))
            painter.drawText(int(x), int(margin + 30 + chart_h + 2), int(bar_w), 18,
                           QtCore.Qt.AlignCenter, label)

        # 量程显示
        font = QtGui.QFont("Arial", 7)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtGui.QColor(120, 120, 120)))
        range_text = f"±{max_val:.0f}"
        painter.drawText(margin, int(center_y) - 5, 30, 10, QtCore.Qt.AlignLeft, range_text)


class CurveVisualizer:
    """曲线图可视化器 - 5通道时间序列"""

    def __init__(self, channels: int = 5, max_points: int = 1000, x_window: float = 10.0):
        self.channels = channels
        self.max_points = max_points
        self.x_window = x_window  # X轴显示时间窗口（秒）
        self.plot_widget = None

        if not PYQTGRAPH_AVAILABLE:
            self.curves = {}
            self.data_buffers = {}
            return

        self.curves = {}
        self.data_buffers = {ch: [] for ch in range(channels)}
        self.channel_time_buffers = {ch: [] for ch in range(channels)}
        self.colors = CHANNEL_COLORS

    def add_plot(self, channel: int, plot_widget):
        if not PYQTGRAPH_AVAILABLE:
            return
        self.plot_widget = plot_widget
        pen = pg.mkPen(color=self.colors[channel % len(self.colors)], width=2)
        curve = plot_widget.plot(pen=pen, name=f'CH{channel + 1}')
        self.curves[channel] = curve

    def update_channel(self, channel: int, value: float, timestamp: float):
        if channel not in self.data_buffers or channel not in self.channel_time_buffers:
            return

        self.data_buffers[channel].append(value)
        self.channel_time_buffers[channel].append(timestamp)

        if len(self.data_buffers[channel]) > self.max_points:
            self.data_buffers[channel].pop(0)
            self.channel_time_buffers[channel].pop(0)

        if channel in self.curves:
            self.curves[channel].setData(
                self.channel_time_buffers[channel],
                self.data_buffers[channel]
            )

        # 限制X轴显示范围为最近x_window秒
        if self.plot_widget and timestamp > 0:
            self.plot_widget.setXRange(timestamp - self.x_window, timestamp, padding=0.02)

    def clear(self):
        for ch in self.data_buffers:
            self.data_buffers[ch].clear()
            if ch in self.channel_time_buffers:
                self.channel_time_buffers[ch].clear()
