import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

from webgame_utils.extracy_small_region import extract_small_regions
import mss
import time
import cv2
import numpy as np
import os
from configs import config
from springc_utils import *


def save_play_patch():
    with mss.mss() as sct:
        img = sct.grab(config.game_region)
        frame = np.array(img)[:, :, :3]           # 转为numpy
    extracted = extract_small_regions(frame)
    save_img(extracted['play_region'], config.p_sv_play)
    
def save_start_patch():
    with mss.mss() as sct:
        img = sct.grab(config.game_region)
        frame = np.array(img)[:, :, :3]           # 转为numpy
    extracted = extract_small_regions(frame)
    save_img(extracted['start_region'], config.p_sv_start) 
    

if __name__ == '__main__':
    # save_play_patch()
    save_start_patch()