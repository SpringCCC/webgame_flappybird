
from dataclasses import dataclass


# # home
# @dataclass
# class Config():
    
#     play_region     =  {"left": -957, "top": 757, "width": 129, "height": 36}
#     start_region    =  {"left": -805, "top": 756, "width": 162, "height": 39}
#     score_region    =  {"left": -747, "top": -32, "width": 140, "height": 51}
#     score_region    =  {"left": -747, "top": -32, "width": 140, "height": 51}
#     game_region     =  {"left": -1114, "top": -81, "width": 785, "height": 1081}

#     small_regions = {'start_region':start_region, 'play_region':play_region, 'score_region':score_region}
    
#     start_pos = (start_region['left'], start_region['top'])
#     play_pos = (play_region['left'], play_region['top'])
    
#     #path
#     p_asset = r"assets"
#     p_mask = r"mask"
#     p_gamewin = r"gamewin"
#     p_sv_start = r"assets\start.jpg"
#     p_sv_play = r"assets\play.jpg"
    
    
#     fps = 20
    

    
#     #dqn
#     max_buffer = 2000
#     batch_size = 32
#     n_actions = 2 # 0:up 1:down
#     lr = 1e-3
#     min_lr = 1e-4
#     T_max = 1000
#     freq_update_params = 5
#     freq_save_model = 1000
#     is_debug = False







# compamy
@dataclass
class Config():
    
    start_region = {"top": 1578, "left": 5032, "width": 176, "height": 39}
    game_region = {"top": 630, "left": 4677, "width": 885, "height": 1227}
    play_region =  {"top": 1576, "left": 4851, "width": 146, "height": 43}
    score_region = {"top": 686, "left": 5088, "width": 164, "height": 55}

    small_regions = {'start_region':start_region, 'play_region':play_region, 'score_region':score_region}
    
    start_pos = (start_region['left'], start_region['top'])
    play_pos = (play_region['left'], play_region['top'])
    
    #path
    p_asset = r"assets"
    p_mask = r"mask"
    p_gamewin = r"gamewin"
    p_sv_start = r"assets\start.jpg"
    p_sv_play = r"assets\play.jpg"
    
    
    fps = 5
    
    resize_scale = 2
    resize_h = game_region['height'] // resize_scale
    resize_w = game_region['width'] // resize_scale
    
    #dqn
    max_buffer = 2000
    batch_size = 32
    n_actions = 2 # 0:up 1:down
    lr = 1e-3
    min_lr = 1e-4
    T_max = 1000
    freq_update_params = 5
    freq_save_model = 1000
    is_debug = False
    gamma = 0.99



# # GS
# @dataclass
# class Config():
#     start_reigon = {"top": 1578, "left": 5032, "width": 176, "height": 39}
#     stop_region ={"top": 1615, "left": 4897, "width": 49, "height": 58}
#     game_region = {"top": 630, "left": 4677, "width": 885, "height": 1227}
#     play_region =  {"top": 1576, "left": 4851, "width": 146, "height": 43}
#     score_region = {"top": 686, "left": 5088, "width": 164, "height": 55}

#     small_regions = {'start_reigon':start_reigon, 'stop_region':stop_region, 'play_region':play_region, 'score_region':score_region}
    
#     start_pos = (start_reigon['left'], start_reigon['top'])
#     play_pos = (play_region['left'], play_region['top'])
    
#     #path
#     p_asset = r"assets"
#     p_mask = r"mask"
#     p_gamewin = r"gamewin"
#     p_sv_start = r"assets\start.jpg"
#     p_sv_play = r"assets\play.jpg"
    
    
#     fps = 20
    

    
#     #dqn
#     max_buffer = 2000
#     batch_size = 32
#     n_actions = 2 # 0:up 1:down
#     lr = 1e-3
#     min_lr = 1e-4
#     T_max = 1000
#     freq_update_params = 5
#     freq_save_model = 1000
#     is_debug = True
    
    
config = Config()