
import os

video_class_name = ['GettingUp', 'KickingBedBoard', 'LiftingLeg', 'NormalGesture', 'PullingOutTubesInMouth',
                    'PullingOutTubesOnBothHands', 'ShakingHead', 'SlappingFence', 'WrigglingBody']
print(len(video_class_name), "类")

for i in range(len(video_class_name)):
    filepath = "C:/Users/24372/Desktop/C3D-Action-Recognition-master/datasets/ICU9/" + video_class_name[i]

    for _, _, video in os.walk(filepath):
        # print(video)
        # 7:3的比例分割
        n = int(len(video) * (7/10))
        # print(n)
        txt_train_file = open('C:/Users/24372/Desktop/C3D-Action-Recognition-master/ucfTrainTestlist/train_file.txt',
                              mode='a')
        txt_test_file = open('C:/Users/24372/Desktop/C3D-Action-Recognition-master/ucfTrainTestlist/test_file.txt',
                             mode='a')
        for j in range(n):
            txt_train_file.write(video_class_name[i] + "_" + str(j+1) + " " + str(i) + "\n")

        for m in range(n, len(video)):
            txt_test_file.write(video_class_name[i] + "_" + str(m+1) + " " + str(i) + "\n")

print("训练集：测试集 = 7:3 划分完毕！")