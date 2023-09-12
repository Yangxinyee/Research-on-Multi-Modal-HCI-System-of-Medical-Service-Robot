# coding=utf8
from models import c3d_model
from keras.optimizers import SGD
import numpy as np
import cv2
import os
import configparser
from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation
from keras.regularizers import l2
from keras.models import Model
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def c3d_model():
    input_shape = (112,112,16,3)
    weight_decay = 0.005
    nb_classes = 9

    inputs = Input(input_shape)
    x = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPool3D((2,2,1),strides=(2,2,1),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes,kernel_regularizer=l2(weight_decay))(x)
    x = Activation('softmax')(x)

    model = Model(inputs, x)
    return model

def main(video_stream):

    # read config.txt
    root_dir=os.path.abspath(os.path.dirname(__file__)) #获取当前文件所在的目录
    configpath = os.path.join(root_dir, "config.txt")
    config = configparser.ConfigParser()
    config.read(configpath)
    classInd_path = config.get("C3D", "classInd_path")
    weights_path = config.get("C3D", "weights_path")
    lr = config.get("C3D", "lr")
    momentum = config.get("C3D", "momentum")
    image_read = config.get("image", "image_read")
    image_write = config.get("image", "image_write")
    video_image = config.get("choose", "video_image")
    with open(classInd_path, 'r') as f:
        class_names = f.readlines()
        f.close()

    # init model
    num = 1
    camera_ids =video_stream.keys()
    print(camera_ids)
    cap_write ={}
    model = c3d_model()
    #sgd = SGD(lr=float(lr), momentum=float(momentum), nesterov=True)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #model.summary()
    model.load_weights(weights_path, by_name=True)
    model = load_model('C:/Users/24372/Desktop/C3D-Action-Recognition-master/results/weights_c3d.h5')

    def multi_detecion(clip, frame):
        inputs = np.array(clip).astype(np.float32)
        inputs = np.expand_dims(inputs, axis=0)
        inputs[..., 0] -= 99.9
        inputs[..., 1] -= 92.1
        inputs[..., 2] -= 82.6
        inputs[..., 0] /= 65.8
        inputs[..., 1] /= 62.3
        inputs[..., 2] /= 60.3
        inputs = inputs[:, :, 8:120, 30:142, :]
        inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
        pred = model.predict(inputs)
        label = np.argmax(pred[0])
        cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 1)
        cv2.putText(frame, "prob: %.4f" % pred[0][label], (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 1)
        clip.pop(0)
        return (frame)

    for i in camera_ids:
        cap_write['cap_'+i] =cv2.VideoCapture(video_stream[i][1])
        size_1 = (int(cap_write['cap_'+i].get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_write['cap_'+i].get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps_1 = cap_write['cap_'+i].get(cv2.CAP_PROP_FPS)
        cap_write["write_" + i]= cv2.VideoWriter(video_stream[i][2], cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps_1, size_1)

    if video_image == 'video':
        while True:
            if num % 2 == 0:
                camera = 'camera_1'
            else:
                camera = 'camera_2'
            ret_1, frame_1 = cap_write['cap_'+str(camera)].read()
            if ret_1:
                tmp = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
                video_stream[camera][0].append(cv2.resize(tmp, (171, 128)))
                if len(video_stream[camera][0]) == 16:
                    frame_1 = multi_detecion(video_stream[camera][0], frame_1)
                    print("16")
                    cap_write['write_'+str(camera)].write(frame_1)
                print (camera+"success")
            num = num + 1
    elif video_image == 'image':
        fileList = os.listdir(image_read)
        fileList.reverse()
        clip = []
        for fileName in fileList:
            frame = cv2.imread(image_read + '/' + fileName)
            clip.append(cv2.resize(frame, (171, 128)))
            if len(clip) == 16:
                frame = multi_detecion(clip, frame)
            cv2.imwrite(image_write + '/' + str(num) + ".jpg", frame)
            print("write success")
            num = num+1
    else:
        print("choose image or video")
    #for i in camera_ids:
     #   cap_write['cap_' + i].release()
     #   print('release'+i)

if __name__ == '__main__':
    video_stream = {'camera_1': [], 'camera_2': [[],'C:/Users/24372/Desktop/ICU9/ShakingHead/ShakingHead_9.mp4','results/ShakingHead_test.mp4' ]}
    video_stream['camera_1'].append([])
    video_stream['camera_1'].append('C:/Users/24372/Desktop/ICU9/GettingUp/GettingUp_9.mp4')
    video_stream['camera_1'].append('results/GettingUp_test.mp4')
    #print(video_stream)
    main(video_stream)