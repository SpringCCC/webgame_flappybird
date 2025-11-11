import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

import win32api
import win32con
from configs import config

def click_at(P):
    """
    移动鼠标到屏幕坐标 (x, y)，点击一次左键
    """
    # 移动鼠标
    win32api.SetCursorPos(P)
    
    # 按下左键
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    
    # 松开左键
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)


def click_start_page():
    click_at(config.start_pos)


def click_play_page():
    click_at(config.play_pos)

def click_action(a):
    if a == 0:
        click_play_page()