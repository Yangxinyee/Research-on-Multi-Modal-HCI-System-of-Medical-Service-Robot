# -*- coding = utf-8 -*-
# @Time : 2023/7/6 18:19
# @Author : Xinye Yang
# @File : gif.py
# @Software : PyCharm

import imageio

# 读取视频文件
video = imageio.get_reader('C:/Users/24372/Desktop/C3D-Action-Recognition-master/datasets/ICU9/GettingUp/GettingUp_3.mp4')

# 创建 gif 动图的写入器
gif_writer = imageio.get_writer('C:/Users/24372/Desktop/gif/GettingUp.gif', mode='I')

# 将视频帧逐一写入 gif 动图中
for frame in video:
    gif_writer.append_data(frame)

# 关闭写入器
gif_writer.close()
