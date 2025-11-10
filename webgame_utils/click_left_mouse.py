import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

import win32api
import win32con
from configs import config

def click_at(x, y):
    """
    移动鼠标到屏幕坐标 (x, y)，点击一次左键
    """
    # 移动鼠标
    win32api.SetCursorPos((x, y))
    
    # 按下左键
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    
    # 松开左键
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

# 示例：点击屏幕坐标 (800, 400)
click_at(*config.play_pos)
