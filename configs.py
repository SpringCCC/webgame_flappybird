
from dataclasses import dataclass

@dataclass
class Config():
    start_reigon = {"top": 758, "left": -802, "width": 158, "height": 35}
    stop_region = {"top": 792, "left": -911, "width": 35, "height": 57}
    game_region = {"top": -54, "left": -1102, "width": 762, "height": 1038}
    play_region =  {"top": 757, "left": -958, "width": 127, "height": 36}
    
    start_pos = (-722, 780)
    play_pos = (-893, 780)
    
    #path
    p_asset = r"assets"
    
    
    fps = 20
    

    
    #dqn
    max_buffer = 2000
    batch_size = 32
    n_actions = 2 # 0:up 1:down
    
    
config = Config()