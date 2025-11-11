
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

from configs import config
import mss
import time
import cv2
import numpy as np
import os
from configs import config
from springc_utils import *
from webgame_utils.score_stop_detect import ScoreDetector
from webgame_utils.extracy_small_region import extract_small_regions
from webgame_utils.start_play_page_detect import *
from webgame_utils.click_actions import *


class Env():

    def __init__(self, fps=10):
        self.fps = fps
        self.sd = ScoreDetector()
        self.sct = mss.mss()
        self.start_page_img = read_img(config.p_sv_start)
        self.play_page_img = read_img(config.p_sv_play)


# small_regions = {'start_reigon':start_reigon, 'stop_region':stop_region, 'play_region':play_region, 'score_region':score_region}
    def capture(self, region=config.game_region):
        game_img = self.sct.grab(region)
        game_img = np.array(game_img)[:, :, :3]
        extracted_imgs = extract_small_regions(game_img)
        extract_start_img = extracted_imgs['start_reigon']
        extract_play_img = extracted_imgs['play_region']
        extract_score_img = extracted_imgs['score_region']
        is_start_page = images_similar_ssim(self.start_page_img, extract_start_img)
        is_play_page = images_similar_ssim(self.play_page_img, extract_play_img)
        score_changed, is_stop_game = self.sd.analyze(extract_score_img)

        if is_play_page:
            click_play_page()
        if is_start_page:
            click_start_page()
        
        return game_img, is_start_page, is_play_page, is_stop_game, score_changed



# def main():
#     sd = ScoreDetector()
#     region = config.game_region

#     fps = 10  
#     interval = 1 / fps
#     frame_count = 0

#     with mss.mss() as sct:
#         start_time = time.time()
#         while time.time() - start_time < duration:
#             print(f"当前是第{frame_count}张图像")
#             frame_start = time.time()

#             # 截图
#             img = sct.grab(region)
#             frame = np.array(img)[:, :, :3]
#             small_regions_img = extract_small_regions(frame, config.game_region, config.small_regions)
#             score_img = small_regions_img['score_region']
#             sd.analyze(score_img)

#             # 保存图片
#             filename = os.path.join(save_dir, f"frame_{frame_count:04d}.png")
#             cv2.imwrite(filename, score_img)
            
#             frame_count += 1
#             # 控制帧率
#             elapsed = time.time() - frame_start
#             if elapsed < interval:
#                 time.sleep(interval - elapsed)


# main()
    