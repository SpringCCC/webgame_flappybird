import numpy as np

def extract_small_regions(big_img: np.ndarray, big_region: dict, small_regions: dict):
    """
    从大图中提取小区域图像
    
    参数：
        big_img: np.ndarray，大图截图数组
        big_region: dict，大图对应屏幕区域，包含 'top', 'left', 'width', 'height'
        small_regions: dict，键为小区域名字，值为区域字典 {'top', 'left', 'width', 'height'}
    
    返回：
        dict，键为小区域名字，值为对应 numpy 图像
    """
    extracted = {}
    for name, region in small_regions.items():
        # 计算相对坐标
        rel_left = region['left'] - big_region['left']
        rel_top  = region['top']  - big_region['top']
        rel_width  = region['width']
        rel_height = region['height']
        
        # 从大图中切片
        small_img = big_img[rel_top:rel_top+rel_height, rel_left:rel_left+rel_width]
        extracted[name] = small_img
    return extracted