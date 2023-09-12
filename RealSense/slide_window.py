# -*- coding = utf-8 -*-
# @Time : 2023/6/3 15:10
# @Author : Xinye Yang
# @File : slide_window.py
# @Software : PyCharm

import os
import cv2
from PIL import Image

for file_num in range(2, 9):
    print("第", file_num, "类正在处理中...")
    # 帧图片路径
    image_path = f'F:/aRealsense/cut_pic_output/{str(file_num)}/'
    '''
    部分变量定义
    '''
    # 获取图片路径下面的所有图片名称
    image_names = os.listdir(image_path)
    image_temp = []
    space = 90  # 剪裁的帧间隔
    video_length = 5  # 期望的剪裁后视频长度，单位：秒s
    frame_length = video_length * 30  # 期望的剪裁后视频长度，单位：帧的个数
    head = 0  # 每段剪裁视频的起始帧位置
    # 对提取到的图片名称进行排序
    image_names.sort(key=lambda n: int(n[:-4]))
    # 设置写入格式为avi
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # 设置每秒帧数
    fps = 30
    # 读取第一个图片获取大小尺寸，因为需要转换成视频的图片大小尺寸是一样的
    image = Image.open(image_path + image_names[0])
    # print("image_names = ", image_names)
    frame_num = len(image_names)
    '''
    模拟剪裁过程，计算可剪裁出的视频数量
    '''
    out_num = 1
    f = 1
    r = f + frame_length - 1
    for k in range(frame_num):
        if r > frame_num:
            if f - frame_num < 90:
                print("最后一段视频不足3s，已被丢弃")
                break
            else:
                out_num += 1
                last_len = (f - frame_num) * 30
                print("最后一段视频不足5s，其长度为：", last_len, "s 已被保留")
        else:
            out_num += 1
            f += space + 1
            r = f + frame_length - 1
    print("共可剪裁出", out_num, "段视频")
    '''
    开始剪裁视频
    '''
    for i in range(out_num):
        if frame_num - head >= 90:
            image_temp = image_names[head:head+frame_length]
            # print(image_temp)
            # 初始化媒体写入对象
            media_path = f'F://aRealsense//cut_video_output//{str(file_num)}//{i+1}.avi'
            video_writer = cv2.VideoWriter(media_path, fourcc, fps, image.size)
            print("剪裁第", i+1, "段视频中")
            print("...")
            # 遍历图片，将每张图片加入视频当中
            for image_name in image_temp:
                im = cv2.imread(os.path.join(image_path, image_name))
                video_writer.write(im)
                # print(image_name, '合并完成')
            # 释放媒体写入对象
            video_writer.release()
            print('第', i+1, '段视频剪裁完成!')
            print()
        else:
            break
        head += space + 1
    print("第", file_num, "类全部剪裁完成！")
