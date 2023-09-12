# -*- coding = utf-8 -*-
# @Time : 2023/5/31 10:35
# @Author : Xinye Yang
# @File : cut_video2pic.py
# @Software : PyCharm

'''将视频按帧分割成图片'''
import cv2
import argparse
import os


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Process pic')
    parser.add_argument('--input', help='video to process', dest='input', default=None, type=str)
    parser.add_argument('--output', help='pic to store', dest='output', default=None, type=str)
    # input为输入视频的路径 ，output为输出存放图片的路径
    argss = parser.parse_args(['--input', r'F:\aRealsense\rgb_data\0\0_all.mp4',
                              '--output', r'F:\aRealsense\cut_pic_output\0'])
    return argss


def process_video(i_video, o_video):
    cap = cv2.VideoCapture(i_video)
    # i_video为视频文件路径表示打开视频

    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) # 获取视频总帧数
    print("Total frame:", num_frame)

    expand_name = '.jpg'
    if not cap.isOpened():
        print("Please check the path.")

    cnt = 0
    while 1:
        ret, frame = cap.read()
        # cap.read()表示按帧读取视频。ret和frame是获取cap.read()方法的两个返回值
        # 其中，ret是布尔值。如果读取正确，则返回TRUE；如果文件读取到视频最后一帧的下一帧，则返回False
        # frame就是每一帧的图像
        if not ret:
            break
        cnt += 1  # 从1开始计帧数
        cv2.imwrite(os.path.join(o_video, str(cnt) + expand_name), frame)
        print("第", cnt, "帧")

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    print('Called with args:')
    print(args)
    process_video(args.input, args.output)
