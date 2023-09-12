# -*- coding = utf-8 -*-
# @Time : 2023/5/30 14:48
# @Author : Xinye Yang
# @File : get_rgb_video.py
# @Software : PyCharm
import datetime
import pyrealsense2 as rs
import numpy as np
import cv2
import sys


class Realsense(object):
    # realsense相机处理类
    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        self.pipeline.start(self.config)  # 开始连接相机
    def get_frame(self):
        frames = self.pipeline.wait_for_frames()  # 获得frame (包括彩色，深度图)
        # 创建对齐对象
        align_to = rs.stream.color  # rs.align允许我们执行深度帧与其他帧的对齐
        align = rs.align(align_to)  # “align_to”是我们计划对齐深度帧的流类型。
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        colorimage = np.asanyarray(color_frame.get_data())
        return colorimage
    def release(self):
        self.pipeline.stop()


if __name__ == '__main__':
    global video_path
    global wr
    print("按下 空格 开始采集RGB视频")
    print("在采集过程中按下 T 键可结束并保存本次采集")
    print("一次采集完成后再次按下 空格 键可重新开始下一次采集")
    print("若要彻底结束程序，则按下 Q 键")
    # 初始化参数
    fps, w, h = 30, 1280, 720
    # 是否录制的标志位 0为不录制
    flag_V = 0
    while True:
        cam = Realsense(w, h, fps)
        while True:
            color_image = cam.get_frame()  # 读取RGB图像帧
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            key = cv2.waitKey(1)
            # 空格键开始采集
            if key & 0xFF == ord(' '):
                flag_V = 1
                # 视频保存路径
                now = datetime.datetime.now()
                video_path = f'F://aRealsense//rgb_data//0//' \
                             f'{now.year}{now.month}{now.day}-{now.hour}{now.minute}-{now.second}.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频格式
                wr = cv2.VideoWriter(video_path, fourcc, fps, (w, h), isColor=True)
                print('...录制视频中...')
            elif key & 0xFF == ord('t'):
                if flag_V == 1:
                    print(f'视频保存在：{video_path}')
                    flag_V = 0
                # 释放资源保存视频
                wr.release()
                print('...按 空格 键开始下一次录制...')
                break
            # Q键彻底退出程序
            elif key & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                wr.release()
                cam.release()
                if flag_V == 1:
                    print(f'视频保存在：{video_path}')
                print("...程序正在退出...")
                sys.exit()
            # 录制开始
            if flag_V == 1:
                wr.write(color_image)  # 保存RGB图像帧
