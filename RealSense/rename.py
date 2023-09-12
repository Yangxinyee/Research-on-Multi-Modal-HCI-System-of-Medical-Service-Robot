# -*- coding = utf-8 -*-
# @Time : 2023/6/4 15:43
# @Author : Xinye Yang
# @File : rename.py
# @Software : PyCharm

import os


def rename():
    i = 0
    path = r"F:\学习课件\大二下\模式识别\模式识别课程设计\项目工程文件\face\data\faces_ye_zhi_ge"

    filelist = os.listdir(path)   # 该文件夹下所有的文件（包括文件夹）
    for files in filelist:   # 遍历所有文件
        i = i + 1
        Olddir = os.path.join(path, files)    # 原来的文件路径
        if os.path.isdir(Olddir):       # 如果是文件夹则跳过
            continue
        filetype = '.jpg'        # 文件扩展名
        Newdir = os.path.join(path, str(i) + filetype)   # 新的文件路径
        os.rename(Olddir, Newdir)    # 重命名
    return True


if __name__ == '__main__':
    rename()

