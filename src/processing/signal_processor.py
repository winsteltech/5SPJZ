#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理和滤波模块 - 5通道圆形电容感应
"""

import numpy as np
from collections import deque
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class DataFilter:
    """数据滤波器 - 支持多种滤波算法"""

    @staticmethod
    def moving_average(data: List[float], window_size: int = 5) -> List[float]:
        if len(data) < window_size:
            return data
        result = []
        for i in range(len(data)):
            if i < window_size - 1:
                result.append(np.mean(data[:i + 1]))
            else:
                result.append(np.mean(data[i - window_size + 1:i + 1]))
        return result

    @staticmethod
    def exponential_smoothing(data: List[float], alpha: float = 0.3) -> List[float]:
        if not data:
            return []
        result = [data[0]]
        for i in range(1, len(data)):
            smoothed = alpha * data[i] + (1 - alpha) * result[-1]
            result.append(smoothed)
        return result

    @staticmethod
    def kalman_filter(data: List[float], process_variance: float = 1e-5,
                     measurement_variance: float = 0.01) -> List[float]:
        result = []
        x = data[0]
        p = 1.0
        for measurement in data:
            p_pred = p + process_variance
            k = p_pred / (p_pred + measurement_variance)
            x = x + k * (measurement - x)
            p = (1 - k) * p_pred
            result.append(x)
        return result

    @staticmethod
    def median_filter(data: List[float], window_size: int = 5) -> List[float]:
        if len(data) < window_size:
            return data
        result = []
        half_window = window_size // 2
        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            result.append(np.median(data[start:end]))
        return result


class StreamingFilter:
    """流式滤波器 - 实时处理流数据"""

    def __init__(self, algorithm: str = 'exponential', window_size: int = 5):
        self.algorithm = algorithm
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.alpha = 0.3
        self.last_smoothed = None
        self.kalman_x = None
        self.kalman_p = 1.0

    def filter(self, value: float) -> float:
        if self.algorithm == 'exponential':
            if self.last_smoothed is None:
                self.last_smoothed = value
            else:
                self.last_smoothed = self.alpha * value + (1 - self.alpha) * self.last_smoothed
            return self.last_smoothed
        elif self.algorithm == 'moving_average':
            self.history.append(value)
            return np.mean(list(self.history))
        elif self.algorithm == 'median':
            self.history.append(value)
            return np.median(list(self.history))
        else:
            return value

    def reset(self):
        self.history.clear()
        self.last_smoothed = None


class SignalProcessor:
    """信号处理器 - 5通道圆形电容感应"""

    def __init__(self, channels: int = 5):
        self.channels = channels
        self.filters = {
            i: StreamingFilter(algorithm='exponential')
            for i in range(channels)
        }

    def process_raw_data(self, raw_data: dict) -> dict:
        """处理原始数据"""
        filtered_data = {}
        for ch, value in raw_data.items():
            filtered_data[ch] = self.filters[ch].filter(float(value))
        return filtered_data

    def compute_statistics(self, channel_data: dict) -> dict:
        """计算统计信息"""
        values = list(channel_data.values())
        if not values:
            return {}
        return {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values))
        }
