# -*- mode: python ; coding: utf-8 -*-

import sys
sys.setrecursionlimit(sys.getrecursionlimit()*5)

block_cipher = None


a = Analysis(['layout.py'],
             pathex=['C:\\Users\\swann\\Documents\\ch3a\\spectrometer'],
             binaries=[],
             datas=[
                 ('assets/my.css', 'assets'),
                 ('assets/new.css', 'assets'),
                 ('assets/spectrumCrop.jpg', 'assets'),
                 ('assets/s3 logo.png', 'assets'),
                 ('C:\\Users\\swann\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\dash_core_components\\', 'dash_core_components'),
                 ('C:\\Users\\swann\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\dash_html_components\\', 'dash_html_components'),
                 ('C:\\Users\\swann\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\dash_renderer\\','dash_renderer'),
				 ('C:\\Users\\swann\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\dash_table\\','dash_table'),
				 ('C:\\Users\\swann\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\dash\\','dash'),
				 ('C:\\Users\\swann\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\flask\\','flask'),
				 ('C:\\Users\\swann\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\plotly\\', 'plotly')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='layout',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='layout')
