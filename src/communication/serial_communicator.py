#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
串口通信模块 - 5通道圆形电容感应
支持 CSV 格式数据解析: "val0,val1,val2,val3,val4\n"
"""

import serial
import threading
import queue
import time
from datetime import datetime
from typing import Callable, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SerialCommunicator:
    """
    串口通信器 - 处理5通道CSV格式数据收发、解析
    数据格式: "1578,1209,1242,1440,1237\n"
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 1.0,
        channels: int = 5
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.channels = channels

        self.serial = None
        self.is_connected = False

        self.raw_data_queue = queue.Queue(maxsize=2000)
        self.data_queue = queue.Queue(maxsize=2000)
        self.receive_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.error_callback: Optional[Callable[[str], None]] = None

        self.receive_thread: Optional[threading.Thread] = None
        self.parse_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self.stats = {
            'packets_received': 0,
            'packets_error': 0,
            'bytes_received': 0,
            'raw_data_blocks': 0,
            'connection_time': None
        }

    def connect(self) -> bool:
        """建立串口连接"""
        try:
            print(f"[串口] 尝试连接到 {self.port}，波特率 {self.baudrate}")

            # 确保serial模块已正确导入
            import serial as serial_module

            self.serial = serial_module.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial_module.EIGHTBITS,
                parity=serial_module.PARITY_NONE,
                stopbits=serial_module.STOPBITS_ONE
            )

            if self.serial.is_open:
                print(f"[串口] 成功打开 {self.port}")
                self.is_connected = True
                self.stats['connection_time'] = datetime.now()

                self.stop_event.clear()
                self.receive_thread = threading.Thread(
                    target=self._receive_loop, daemon=True
                )
                self.receive_thread.start()

                self.parse_thread = threading.Thread(
                    target=self._parse_loop, daemon=True
                )
                self.parse_thread.start()

                logger.info(f"已连接到 {self.port}，波特率 {self.baudrate}")
                return True
            else:
                logger.error(f"串口 {self.port} 未成功打开")
                if self.error_callback:
                    self.error_callback(f"串口 {self.port} 未成功打开")
                return False

        except PermissionError as e:
            error_msg = f"串口连接失败: 端口 {self.port} 被占用或无访问权限 ({e})".strip()
            logger.error(error_msg)
            print(f"[串口错误] {error_msg}")
            try:
                import serial.tools.list_ports
                available_ports = serial.tools.list_ports.comports()
                port_list = [p.device for p in available_ports]
                print(f"[串口] 可用串口列表: {port_list}")
            except Exception:
                pass
            if self.error_callback:
                self.error_callback(error_msg)
            return False

        except Exception as e:
            error_msg = f"串口连接失败: {e}"
            logger.error(error_msg)
            print(f"[串口错误] {error_msg}")
            if self.error_callback:
                self.error_callback(error_msg)
            return False

    def disconnect(self):
        """断开连接"""
        self.stop_event.set()
        if self.receive_thread:
            self.receive_thread.join(timeout=2)
        if self.parse_thread:
            self.parse_thread.join(timeout=2)
        if self.serial and self.serial.is_open:
            self.serial.close()
            self.is_connected = False
            logger.info("串口已断开")

    def send_command(self, data: bytes) -> bool:
        """发送命令"""
        if not self.is_connected or not self.serial:
            return False
        try:
            self.serial.write(data)
            return True
        except Exception as e:
            logger.error(f"发送失败: {e}")
            if self.error_callback:
                self.error_callback(f"发送失败: {e}")
            return False

    def _receive_loop(self):
        """接收循环 - 采集线程，按换行符分割数据块"""
        buffer = b''
        while not self.stop_event.is_set():
            try:
                if self.serial and self.serial.in_waiting:
                    data = self.serial.read(self.serial.in_waiting)
                    if data:
                        buffer += data
                        self.stats['bytes_received'] += len(data)

                        while b'\n' in buffer:
                            newline_pos = buffer.find(b'\n')
                            if newline_pos != -1:
                                data_block = buffer[:newline_pos + 1]
                                buffer = buffer[newline_pos + 1:]
                                try:
                                    self.raw_data_queue.put_nowait(data_block)
                                    self.stats['raw_data_blocks'] += 1
                                except queue.Full:
                                    logger.warning("原始数据队列已满，丢弃数据块")
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"接收错误: {e}")
                if self.error_callback:
                    self.error_callback(f"接收错误: {e}")
                buffer = b''

    def _parse_loop(self):
        """解析循环 - 解析CSV格式数据块"""
        while not self.stop_event.is_set():
            try:
                data_block = self.raw_data_queue.get(timeout=0.1)
                if data_block:
                    packets = self._parse_data(data_block)
                    if packets:
                        for packet in packets:
                            try:
                                self.data_queue.put_nowait(packet)
                                self.stats['packets_received'] += 1
                                if self.receive_callback:
                                    self.receive_callback(packet)
                            except queue.Full:
                                logger.warning("数据队列已满，丢弃解析后的数据")
                                self.stats['packets_error'] += 1
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"解析错误: {e}")
                self.stats['packets_error'] += 1
                if self.error_callback:
                    self.error_callback(f"解析错误: {e}")

    def _parse_data(self, buffer: bytes) -> list:
        """
        解析CSV格式数据: "val0,val1,val2,val3,val4\n"
        """
        packets = []

        try:
            text_data = buffer.decode('utf-8', errors='ignore').strip()

            if any(ord(c) < 32 and c not in '\r\n' for c in text_data):
                return packets

            # 去除可能的日志前缀，如 "[19:18:43.638]收←◆"
            if '◆' in text_data:
                idx = text_data.find('◆')
                text_data = text_data[idx + 1:].strip()

            if ',' in text_data:
                values = text_data.split(',')
                if len(values) >= self.channels:
                    try:
                        channel_values = [float(v.strip()) for v in values[:self.channels]]
                        packet = {
                            'timestamp': datetime.now().isoformat(),
                            'channels': {
                                i: channel_values[i]
                                for i in range(self.channels)
                            }
                        }
                        packets.append(packet)
                        return packets
                    except ValueError as e:
                        print(f"[解析] 数值转换错误: {e}")
        except UnicodeDecodeError:
            pass

        return packets

    def get_data(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """获取接收到的数据"""
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """获取通信统计"""
        return self.stats.copy()

    def clear_queue(self):
        """清空数据队列"""
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
