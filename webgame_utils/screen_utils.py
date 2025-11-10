import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)


from skimage.metrics import structural_similarity as ssim

import mss
import numpy as np
import cv2
import time
from configs import config
from springc_utils import *

sct = mss.mss()

def continue_grab_region_image(region, fps):
# 目标帧率
    frame_count = 0
    start_time = time.time()
    cnt = 0
    p = r"D:\code\rl\webgame\web_game_tools_demo\gamewin"
    checkroot(p)
    img_stop = read_img(r"assets\stop.jpg")
    img_stop_gray = cv2.cvtColor(img_stop, cv2.COLOR_BGR2GRAY)
    threshold=0.9
    while True:
        frame_start = time.time()
        cnt += 1
        # 截取画面
        img = np.array(sct.grab(config.stop_region))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(img_stop_gray, gray_img, full=True)
        frame_count += 1
        save_img(img, f"{p}\{cnt}.jpg")
        print(f"{frame_count = }\t {score = }")
        if score > threshold:
            break
        
        elapsed = time.time() - frame_start
        sleep_time = max(0, interval - elapsed)
        time.sleep(sleep_time)


    cv2.destroyAllWindows()


continue_grab_region_image()