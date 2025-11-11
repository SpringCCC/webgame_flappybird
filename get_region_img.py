import mss
import numpy as np
import time
from springc_utils import *
from configs import config


# 初始化 mss

def get_region_img(region, sv_img_name):
    sct = mss.mss()
    img = np.array(sct.grab(region))
    save_img(img, f"{config.p_asset}\{sv_img_name}")

# get_region_img(config.game_region, "game.jpg")
# get_region_img(config.stop_region, "stop.jpg")
get_region_img(config.start_region, "start.jpg")
# get_region_img(config.play_region, "play.jpg")