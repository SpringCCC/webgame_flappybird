from pynput import mouse

coords = []


##运行后，点击可以显示鼠标位置，用于提取屏幕指定位置的信息

def on_click(x, y, button, pressed):
    if pressed:
        print(f"鼠标点击位置: ({x}, {y})")
        coords.append((x, y))
    if len(coords) == 2:  # 点击两次获取左上角和右下角
        # 计算截图区域
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        left = min(x1, x2)
        top = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        print(f"游戏区域: left={left}, top={top}, width={width}, height={height}")
        # 停止监听
        return False

# 开始监听鼠标点击
with mouse.Listener(on_click=on_click) as listener:
    listener.join()