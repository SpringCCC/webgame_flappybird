# import sys
# import os
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, parent_dir)
# os.chdir(parent_dir)

# import mss
# import time
# import cv2
# import numpy as np
# import os
# from configs import config
# from springc_utils import *


# region = config.game_region
# # 输出文件夹
# save_dir = "gamewin"
# # checkroot(save_dir)

# fps = 10  # 每秒截10帧
# interval = 1 / fps
# frame_count = 0
# duration = 100  # 录制时间，单位秒（你可改为任意时长）

# with mss.mss() as sct:
#     start_time = time.time()
#     while time.time() - start_time < duration:
#         frame_start = time.time()

#         # 截图
#         img = sct.grab(region)
#         frame = np.array(img)[:, :, :3]           # 转为numpy
#         # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # 保存图片
#         filename = os.path.join(save_dir, f"frame_{frame_count:04d}.png")
#         cv2.imwrite(filename, frame)
        
#         frame_count += 1

#         # 控制帧率
#         elapsed = time.time() - frame_start
#         if elapsed < interval:
#             time.sleep(interval - elapsed)

# print(f"✅ 截取完成，共保存 {frame_count} 帧到 {save_dir}/ 文件夹中。")
