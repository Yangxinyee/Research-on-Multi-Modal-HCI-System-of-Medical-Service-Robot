# -*- coding = utf-8 -*-
# @Time : 2023/5/31 10:44
# @Author : Xinye Yang
# @File : trans_pic2video.py
# @Software : PyCharm
# -*- coding: utf-8 -*-

import os
import cv2
from PIL import Image
'''
思路：
视频已经被按帧数分割成图片，视频有几帧，图片就有几张，这样可以对于帧进行直观操作。
实现的功能：按照人为设定的帧数间隔，将帧图片合成5s一段的视频，由于视频是30fps，故5s视频共包含150帧图片。
'''
'''
算法流程：
起先的150帧被拼接为第一段5s视频，随后滑动窗口（滑动距离为人为设置的帧间隔space）
第二段5s视频的起始帧变为第space+1+1帧
第三段5s视频的起始帧变为第space+1+space+1+1帧
以此类推，第n段视频的起始帧变为第(n-1)*space+n帧
若最后剩下的帧数不足90帧(3s)则舍弃掉，否则保留。
'''
'''
算法变量定义：
图片名字被保存在一个列表image_names内(升序排列1，2，3...)，长度为图片数量保存在frame_num变量
设置一个头指针head作为滑动头，space为设置的帧间隔
临时图片缓存列表image_temp储存每一段5s视频的150张图片名字。
'''
'''
算法实现：
获取帧数量，存入frame_num
head = 0，space = 30

'''
'''
图片合成视频函数
:param image_path: 图片路径
:param media_path: 合成视频保存路径
:return:
'''
def image_to_video(image_path, media_path):

    # 获取图片路径下面的所有图片名称
    image_names = os.listdir(image_path)
    # 对提取到的图片名称进行排序
    image_names.sort(key=lambda n: int(n[:-4]))
    # 设置写入格式
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # 设置每秒帧数
    fps = 30
    # 读取第一个图片获取大小尺寸，因为需要转换成视频的图片大小尺寸是一样的
    image = Image.open(image_path + image_names[0])
    # 初始化媒体写入对象
    video_writer = cv2.VideoWriter(media_path, fourcc, fps, image.size)
    # 遍历图片，将每张图片加入视频当中
    for image_name in image_names:
        im = cv2.imread(os.path.join(image_path, image_name))
        video_writer.write(im)
        print(image_name, '合并完成')
    # 释放媒体写入对象
    video_writer.release()
    print('第', '段视频写入完成')


# 图片路径
image_path = 'F:/aRealsense/cut_pic_output/'
# 视频保存路径+名称
media_path = "F:/aRealsense/cut_video_output/out.avi"
# 调用函数，生成视频
image_to_video(image_path, media_path)
