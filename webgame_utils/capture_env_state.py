
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
        game_img = np.array(game_img)[:, :, :3] # BGR
        extracted_imgs = extract_small_regions(game_img)
        extract_start_img = extracted_imgs['start_region']
        extract_play_img = extracted_imgs['play_region']
        extract_score_img = extracted_imgs['score_region']
        is_start_page = images_similar_ssim(self.start_page_img, extract_start_img)
        is_play_page = images_similar_ssim(self.play_page_img, extract_play_img)
        score_changed, is_stop_game = self.sd.analyze(extract_score_img, game_img)

        if is_play_page:
            click_play_page()
        if is_start_page:
            click_start_page()
        print(f"{is_start_page}\t{is_play_page=}\t{is_stop_game=}\t{score_changed=}")
        return game_img, is_start_page, is_play_page, is_stop_game, score_changed



if __name__ == '__main__':
    env = Env()
    game_img, is_start_page, is_play_page, is_stop_game, score_changed = env.capture()
    print(is_start_page, is_play_page, is_stop_game, score_changed)