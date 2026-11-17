#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据API服务器模块
提供HTTP REST接口，发送传感器原始数据和力学计算数据
"""

import json
import time
import threading
import logging
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from flask import Flask, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logging.warning("Flask未安装，API功能不可用。请运行: pip install flask")

logger = logging.getLogger(__name__)


class DataAPIServer:
    """
    数据API服务器
    
    提供两个数据端点：
    1. /api/raw - 传感器原始数据（5通道电容值）
    2. /api/force - 力学计算数据（总压力、剪切力X/Y、质心等）
    
    启动后监听指定端口，外部可通过HTTP GET获取最新数据
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        self.host = host
        self.port = port
        self.app: Optional[Flask] = None
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False

        # 数据存储（线程安全）
        self._lock = threading.Lock()
        self._raw_data: Dict[str, Any] = {}
        self._force_data: Dict[str, Any] = {}
        self._last_update_time = 0.0

        if not FLASK_AVAILABLE:
            logger.error("Flask未安装，API服务器无法启动")
            return

        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        """设置API路由"""

        @self.app.route('/api/raw', methods=['GET'])
        def get_raw_data():
            """获取传感器原始数据"""
            with self._lock:
                response = {
                    'timestamp': datetime.now().isoformat(),
                    'channels': self._raw_data.copy(),
                    'update_time': self._last_update_time
                }
            return jsonify(response)

        @self.app.route('/api/force', methods=['GET'])
        def get_force_data():
            """获取力学计算数据"""
            with self._lock:
                response = {
                    'timestamp': datetime.now().isoformat(),
                    'force': self._force_data.copy(),
                    'update_time': self._last_update_time
                }
            return jsonify(response)

        @self.app.route('/api/all', methods=['GET'])
        def get_all_data():
            """获取所有数据（原始+力学）"""
            with self._lock:
                response = {
                    'timestamp': datetime.now().isoformat(),
                    'raw': self._raw_data.copy(),
                    'force': self._force_data.copy(),
                    'update_time': self._last_update_time
                }
            return jsonify(response)

        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            """获取API服务器状态"""
            return jsonify({
                'status': 'running',
                'port': self.port,
                'last_update': self._last_update_time,
                'has_raw_data': bool(self._raw_data),
                'has_force_data': bool(self._force_data)
            })

        @self.app.route('/', methods=['GET'])
        def index():
            """API根路径，返回可用端点说明"""
            return jsonify({
                'name': '5通道电容感应传感器数据API',
                'version': 'v1.3',
                'endpoints': {
                    '/api/raw': '传感器原始数据（5通道电容值）',
                    '/api/force': '力学计算数据（总压力、剪切力等）',
                    '/api/all': '所有数据（原始+力学）',
                    '/api/status': 'API服务器状态'
                }
            })

    def update_raw_data(self, channel_data: Dict[int, float]):
        """更新原始传感器数据"""
        with self._lock:
            self._raw_data = {f'ch{i+1}': float(v) for i, v in channel_data.items()}
            self._last_update_time = time.time()

    def update_force_data(self, total_pressure: float, shear_x: float, shear_y: float,
                          center_x: float = 0.0, center_y: float = 0.0,
                          radius: float = 0.0):
        """更新力学计算数据"""
        with self._lock:
            self._force_data = {
                'total_pressure': float(total_pressure),
                'shear_x': float(shear_x),
                'shear_y': float(shear_y),
                'shear_magnitude': float((shear_x**2 + shear_y**2)**0.5),
                'center_x': float(center_x),
                'center_y': float(center_y),
                'equivalent_radius': float(radius)
            }
            self._last_update_time = time.time()

    def start(self):
        """启动API服务器（在后台线程中运行）"""
        if not FLASK_AVAILABLE:
            logger.error("Flask未安装，无法启动API服务器")
            return False

        if self.is_running:
            logger.warning("API服务器已在运行")
            return True

        self.is_running = True
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self.server_thread.start()
        logger.info(f"API服务器已启动: http://{self.host}:{self.port}")
        return True

    def _run_server(self):
        """运行Flask服务器"""
        try:
            # 关闭Flask默认日志，避免输出过多
            import logging as _logging
            _logging.getLogger('werkzeug').setLevel(_logging.ERROR)
            self.app.logger.setLevel(_logging.ERROR)

            self.app.run(
                host=self.host,
                port=self.port,
                threaded=True,
                debug=False,
                use_reloader=False
            )
        except Exception as e:
            logger.error(f"API服务器运行错误: {e}")
            self.is_running = False

    def stop(self):
        """停止API服务器"""
        self.is_running = False
        # Flask没有内置的优雅停止方法，依赖线程daemon属性
        logger.info("API服务器已停止")

    def is_alive(self) -> bool:
        """检查服务器是否运行中"""
        return self.is_running and self.server_thread is not None and self.server_thread.is_alive()
