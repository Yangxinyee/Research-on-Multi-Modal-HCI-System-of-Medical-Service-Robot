# -*- coding = utf-8 -*-
# @Time : 2023/7/4 15:51
# @Author : Xinye Yang
# @File : resize.py
# @Software : PyCharm

from moviepy.editor import *
namelist = ['KickingBedBoard','LiftingHands','LiftingLegs','Normal','WrigglingBody']

for i in range(len(namelist)):
    SourcePath = "C:\\Users\\24372\\Desktop\\ICU1\\" + namelist[i]
    savepath = "C:\\Users\\24372\\Desktop\\ICU\\" + namelist[i]
    FileList = []
    FileName = []
    for a, b, c in os.walk(SourcePath):
        for name in c:
            fname = os.path.join(a, name)
            if fname.endswith(".mp4"):
                FileList.append(fname)
                FileName.append(name)
    print(FileList)
    print(FileName)

    for i in range(len(FileList)):
        clip = VideoFileClip(FileList[i])
        clip = clip.resize((480, 270))
        OutputName = os.path.join(savepath, FileName[i][:-4]+'.mp4')
        clip.write_videofile(OutputName, bitrate='25000000', threads=1)
