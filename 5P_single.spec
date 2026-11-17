#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyInstaller 打包配置 - 单文件 EXE 模式

使用方法（在项目根目录执行）：
    pip install pyinstaller
    pyinstaller 5P_single.spec

产物输出到 dist/5P_v1.6.exe
"""
import os

block_cipher = None

a = Analysis(
    ['src/main.py'],
    # 使用相对路径，避免硬编码本机绝对路径
    pathex=[os.path.abspath('.')],
    binaries=[],
    datas=[],
    hiddenimports=[
        'PyQt5',
        'pyqtgraph',
        'numpy',
        'serial',
        'serial.tools',
        'serial.tools.list_ports',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 排除未使用的大体积库以缩减体积
    excludes=['matplotlib', 'scipy', 'pandas', 'tkinter'],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='5P_v1.6',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon=None,
)
