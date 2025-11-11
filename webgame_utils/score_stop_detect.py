import mss
import numpy as np
import pytesseract
import cv2
from springc_utils import *
from configs import config


import cv2
import numpy as np


class ScoreDetector:
    def __init__(self, white_thresh=230, diff_thresh=0.9, min_white_ratio=0.005):
        """
        white_thresh: 白色像素阈值 (RGB每通道大于此值认为是白色)
        diff_thresh: 判断数字是否变化的阈值
        min_white_ratio: 小于此比例认为分数消失
        """
        self.prev_mask = None
        self.white_thresh = white_thresh
        self.diff_thresh = diff_thresh
        self.min_white_ratio = min_white_ratio
        self.c = 0

    def get_white_mask(self, img):
        """提取白色区域掩膜"""
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, self.white_thresh, 255)
        return mask

    def analyze(self, img):

        mask = self.get_white_mask(img)

        if config.is_debug:
            vis = img.copy()
            vis[mask > 0] = [0, 0, 255]  # 将白色区域标红
            save_img(vis, f"mask/{self.c}.jpg")
            self.c+=1

        white_ratio = np.sum(mask > 0) / mask.size
        score_disappeared = white_ratio < self.min_white_ratio
        score_changed = False
        if self.prev_mask is not None and not score_disappeared:
            union = np.bitwise_or(self.prev_mask, mask)
            inter = np.bitwise_and(self.prev_mask, mask)
            white_ratio = np.sum(inter > 0) / (np.sum(union > 0)+1)
            score_changed = white_ratio < self.diff_thresh

        self.prev_mask = mask
        return score_changed, score_disappeared
