# -*- coding = utf-8 -*-
# @Time : 2023/7/3 11:34
# @Author : Xinye Yang
# @File : test.py
# @Software : PyCharm

# from mmaction.apis import init_recognizer, inference_recognizer
#
# config_file = 'D:/mmaction2-main/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
# # download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '../mmaction2-main/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'
# # build the model from a config file and a checkpoint file
# model = init_recognizer(config_file, checkpoint_file, device='cpu')
#
# # test a single video and show the result:
# video = '../mmaction2-main/demo/demo.mp4'
# label = '../mmaction2-main/tools/data/kinetics/label_map_k400.txt'
# results = inference_recognizer(model, video)
#
# labels = open(label).readlines()
# labels = [x.strip() for x in labels]
# # results = [(labels[k[0]], k[1]) for k in results]
# print(results)
# # # show the results
# # for result in results:
# #     print(f'{result[0]}: ', result[1])
from collections import deque
result_queue = deque([[('WrigglingBody', 0.003808587556704879), ('SlappingFence', 0.0035701573360711336),
                ('KickingBedBoard', 0.003235325450077653), ('GettingUp', 0.0029384088702499866),
                ('PullingOutTubesOnBothHands', 0.002493527252227068)]], maxlen=1)
print(len(result_queue))
if len(result_queue) != 0:
    text_info = {}
    results = result_queue.popleft()
    print(results)
    for i, result in enumerate(results):
        selected_label, score = result
        if score < 0:
            break
        location = (0, 40 + i * 20)
        text = selected_label + ': ' + str(round(score * 10000, 2)) + ' %'
        text_info[location] = text

print(text_info)