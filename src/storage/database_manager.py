#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库存储模块 - 5通道圆形电容感应
SQLite + 缓存 + 批量插入优化
"""

import sqlite3
import threading
import queue
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

NUM_CHANNELS = 5


class DatabaseManager:
    """高性能数据库管理器 - 5通道"""

    def __init__(self, db_path: str = 'data/sensor_data.db',
                 cache_size: int = 1000,
                 batch_interval: float = 5.0):
        self.db_path = db_path
        self.cache_size = cache_size
        self.batch_interval = batch_interval

        Path(db_path).parent.mkdir(exist_ok=True, parents=True)

        self.cache = []
        self.cache_lock = threading.Lock()
        self.write_queue = queue.Queue()

        self.write_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self._init_database()
        self._start_write_thread()

    def _init_database(self):
        """初始化数据库表"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 构建5通道的列定义
            col_defs = ', '.join([f'channel_{i} REAL' for i in range(NUM_CHANNELS)])
            col_names = ', '.join([f'channel_{i}' for i in range(NUM_CHANNELS)])
            placeholders = ', '.join(['?'] * (NUM_CHANNELS + 1))

            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS sensor_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    {col_defs},
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON sensor_data(timestamp)
            ''')

            conn.commit()
            conn.close()
            logger.info(f"数据库已初始化: {self.db_path}")

        except sqlite3.Error as e:
            logger.error(f"数据库初始化失败: {e}")

    def add_record(self, timestamp: str, channel_data: Dict[int, float]):
        """添加单条记录到缓存"""
        record = {
            'timestamp': timestamp,
            'channels': channel_data,
            'created_at': datetime.now().isoformat()
        }

        with self.cache_lock:
            self.cache.append(record)
            if len(self.cache) >= self.cache_size:
                self._flush_cache()

    def _flush_cache(self):
        if not self.cache:
            return
        try:
            self.write_queue.put_nowait(self.cache.copy())
            self.cache.clear()
        except queue.Full:
            logger.warning("写入队列已满，部分数据可能丢失")

    def _start_write_thread(self):
        self.stop_event.clear()
        self.write_thread = threading.Thread(
            target=self._write_worker, daemon=True
        )
        self.write_thread.start()

    def _write_worker(self):
        last_flush = time.time()
        pending_records = []

        while not self.stop_event.is_set():
            try:
                batch = self.write_queue.get(timeout=1)
                pending_records.extend(batch)
            except queue.Empty:
                if time.time() - last_flush > self.batch_interval:
                    with self.cache_lock:
                        if self.cache:
                            self._flush_cache()

            if pending_records:
                self._batch_insert(pending_records)
                pending_records = []
                last_flush = time.time()

        if pending_records:
            self._batch_insert(pending_records)

    def _batch_insert(self, records: List[Dict[str, Any]]):
        if not records:
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            col_names = ', '.join(['channel_' + str(i) for i in range(NUM_CHANNELS)])
            all_cols = 'timestamp, ' + col_names
            placeholders = ', '.join(['?'] * (NUM_CHANNELS + 1))

            values = []
            for record in records:
                row = [record['timestamp']]
                for ch in range(NUM_CHANNELS):
                    value = record['channels'].get(ch, 0.0)
                    row.append(value)
                values.append(tuple(row))

            cursor.executemany(
                f'INSERT INTO sensor_data ({all_cols}) VALUES ({placeholders})',
                values
            )

            conn.commit()
            conn.close()
            logger.debug(f"已写入 {len(records)} 条记录")

        except sqlite3.Error as e:
            logger.error(f"数据库写入失败: {e}")

    def query_data(self, start_time: Optional[str] = None,
                  end_time: Optional[str] = None,
                  limit: int = 1000) -> List[Dict[str, Any]]:
        """查询数据"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM sensor_data WHERE 1=1"
            params = []

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()

            data = []
            for row in results:
                record = {'timestamp': row['timestamp'], 'channels': {}}
                for ch in range(NUM_CHANNELS):
                    record['channels'][ch] = row[f'channel_{ch}']
                data.append(record)
            return data

        except sqlite3.Error as e:
            logger.error(f"数据库查询失败: {e}")
            return []

    def flush(self):
        with self.cache_lock:
            if self.cache:
                self._flush_cache()

    def close(self):
        self.flush()
        self.stop_event.set()
        if self.write_thread:
            self.write_thread.join(timeout=5)
        logger.info("数据库已关闭")

    def get_row_count(self) -> int:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sensor_data")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except sqlite3.Error as e:
            logger.error(f"获取行数失败: {e}")
            return 0

    def export_to_csv(self, output_file: str, start_time: Optional[str] = None,
                     end_time: Optional[str] = None):
        """导出数据到CSV"""
        import csv
        try:
            data = self.query_data(start_time, end_time, limit=100000)
            if not data:
                logger.warning("没有数据可导出")
                return

            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['timestamp'] + [f'channel_{ch}' for ch in range(NUM_CHANNELS)]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for record in data:
                    row = {'timestamp': record['timestamp']}
                    row.update({f'channel_{ch}': record['channels'][ch]
                               for ch in range(NUM_CHANNELS)})
                    writer.writerow(row)

            logger.info(f"已导出 {len(data)} 条记录到 {output_file}")

        except Exception as e:
            logger.error(f"导出失败: {e}")
