# Research on Multi-Modal HCI System of Medical Service Robot

*The code is still being worked on......*



## Application Scenarios

ICU and other medical scenarios：

<img src="https://i.imgur.com/maBD5Qd.png" width="50%">

Nursing homes and other physical therapy facilities：

<img src="https://i.imgur.com/iScV9rm.png" width="50%">



## Ⅰ Computer Vision Monitoring System Based on Action Recognition Algorithm.

The computer vision monitoring system is mainly developed and deployed based on the mainstream video motion recognition algorithm. Mainstream motion recognition algorithms include C3D, TSN, SlowFast, etc. 

Due to the particularity of the task, it is impossible to use public data sets, such as UCF101 and Kinetics 400, etc., so it is necessary to independently collect data and make standard data sets for model training. 

After the model is trained, it needs to be deployed on a mobile platform for real-time high-frame rate model inference. The mobile platform is **NVIDIA Jetson AGX Orin DEVELOPER KIT**, and the target real-time detection frame rate is more than 20FPS.



### Production of the Dataset

UCF101 and its structure：

<img src="https://i.imgur.com/5FOan7k.png" width="80%">

Realsense Camera：

<img src="https://i.imgur.com/KMhME9a.png" width="30%">

**Self-produced Mock Dataset in ICU---ICU9**

<table>
    <tr>
        <td ><center><img src="https://i.imgur.com/MRCiBkm.gif" width="120%"></center></td>
        <td ><center><img src="https://i.imgur.com/cISBplv.gif" width="120%"></center></td>
        <td ><center><img src="https://i.imgur.com/Lpcc8Yw.gif" width="120%"></center></td>
    </tr>
    <tr>
    <td><center><img src="https://i.imgur.com/Qisd9pj.gif" width="120%"></center></td>
    <td ><center><img src="https://i.imgur.com/MnIZ9QR.gif" width="120%"></center></td>
    <td ><center><img src="https://i.imgur.com/HXHabFK.gif" width="120%"></center></td>
	</tr>
    <tr>
    <td><center><img src="https://i.imgur.com/ovdVXLW.gif" width="120%"></center></td>
    <td><center><img src="https://i.imgur.com/4qtsX9h.gif" width="120%"></center></td>
    <td><center><img src="https://i.imgur.com/smANWl3.gif" width="120%"></center></td>
	</tr>
</table>

**Categories:**

| ShakingHead       | PullingOutTubesOnBothHands | PullingOutTubesInMouth |
| ----------------- | -------------------------- | ---------------------- |
| **LiftingLeg**    | **WrigglingBody**          | **GettingUp**          |
| **NormalGesture** | **KickingBedBoard**        | **SlappingFence**      |

**However, the diversity of simulation data is insufficient, and more information needs to be obtained from limited data.**

In view of this, I propose a video sampling method based on sliding window.

The basic principle is as follows: 5s is used as a fixed window Length (that is, the length of each short video), let this window slide on the captured large video to capture the subvideo, and the sliding interval is adjustable. In this way, more small segments of data can be obtained in the limited video data.

**After all, the simulated data set cannot be compared with the data collected in the real environment, so at this stage, we actively cooperate with the hospital to collect the real data.**

<table>
    <tr>
    <td ><center><img src="https://i.imgur.com/vBkyv4u.gif" width="120%"></center></td>
    <td ><center><img src="https://i.imgur.com/llGWAJO.gif" width="120%"></center></td>
    </tr>
    <tr>
    <td ><center><img src="https://i.imgur.com/H81YhMj.gif" width="120%"></center></td>
    <td ><center><img src="https://i.imgur.com/Ku8BQkQ.gif" width="120%"></center></td>
    </tr>
</table>

***However, real-world data is more complex and noisy, requiring more efficient ways to deal with......***
***The above images have been authorized by the parties.***



### Action Recognition Algorithm

TOOL: MMAction2

<img src="https://i.imgur.com/H6JmuV9.png" width="40%">

#### C3D

For problems based on video analysis, 2D convolution is not good at capturing temporal information, so 3D convolution is proposed to solve this problem.

<img src="https://i.imgur.com/kzrorjq.png" width="90%">

This network I used is quite different from the structure in the paper. 

**The network used for training consists of five convolutional layers, five pooling layers, two fully connected layers, and one softmax output layer**

After experiments, using the model structure in the paper will increase the model parameters and make the model complicated, which makes the learning more difficult. 

**The simplified model structure enables faster and more accurate learning of features in the data.**

```python
def c3d_model():
    input_shape = (480,270,16,3)
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
```

**Results Visualization:**

Use source code to train from scratch, the curve is smooth, and the final loss convergence is rapid. The model training accuracy is high, reaching 90%, but the generalization performance is slightly poor, reaching about 70% accuracy.

Use the mmaction2 toolkit and fine-tune training using the pre-trained model. The curve is relatively smooth, and the final loss convergence is relatively rapid, and the model training accuracy is higher, reaching 95%+.

<table>
    <tr>
    <td ><center><img src="https://i.imgur.com/nRCSuSS.png" width="120%"></center></td>
    <td ><center><img src="https://i.imgur.com/p0HAf2m.png" width="120%"></center></td>
    </tr>
    <tr>
    <td><center><img src="https://i.imgur.com/XXgtrOm.png" width="120%"></center></td>
    <td ><center><img src="https://i.imgur.com/rSeNc0a.png" width="120%"></center></td>
    </tr>
</table>

<div align=center><img src="https://i.imgur.com/7o1mnrx.png" width="60%"></div>

***There's room for improvement......***



#### TSN

An input video is divided into K segments, and a snippet (several frames of images superimposed) is randomly sampled from its corresponding segment. Different segments of the category score fusion. This is a video-level prediction. The predictions of all models are then combined to produce the final prediction.

<img src="https://i.imgur.com/UGkBQgH.png" width="80%">

**Results Visualization:**

The curve is relatively smooth, the final loss convergence is relatively rapid, the model training accuracy is high, up to 99%, the generalization performance is also good, can reach an average accuracy of about 95%. 

At the same time, the heat map is drawn, and it is found that each category is more accurate, and there is no need to reduce the number of categories.

<table>
    <tr>
    <td ><center><img src="https://i.imgur.com/gzgeIMV.png" width="120%"></center></td>
    <td ><center><img src="https://i.imgur.com/7Ri0OeD.png" width="120%"></center></td>
    </tr>
</table>

<div align=center><img src="https://i.imgur.com/s72Y1Fp.png" width="50%"></div>

#### SlowFast

SlowFast is a video recognition paper at Facebook's 2019 ICCV. In this paper, a new SlowFast network architecture is proposed to realize the processing and analysis of time and space dimensions by two branches respectively.

<img src="https://i.imgur.com/TRsVk4n.png" width="80%">

**Results Visualization:**

*The overall effect of the SlowFast model is close to the TSN model, which is omitted here...*



### Demonstration of Real-time Action Recognition

<table>
    <tr>
    <td><center><img src="https://i.imgur.com/9TVvvqx.gif" width="120%"></center></td>
    <td><center><img src="https://i.imgur.com/UrB0DOP.gif" width="120%"></center></td>
    </tr>
    <tr>
    <td><center><img src="https://i.imgur.com/HCRgCJq.gif" width="120%"></center></td>
    <td><center><img src="https://i.imgur.com/05fgeWw.gif" width="120%"></center></td>
    </tr>
</table>



## Ⅱ Simulation of Robot Navigation and Obstacle Avoidance Algorithm in Crowd.

**References:**
***Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning. ICRA 2019.*** 

***Relational Graph Learning for Crowd Navigation. IROS 2020.***  

This task mainly focuses on replicating the algorithms based on graph convolutional neural network and reinforcement learning in the paper, and for this project, adjusting the experimental environment and adding the model distillation method to simulate the crowd navigation and obstacle avoidance of robots.

<div align=center><img src="https://i.imgur.com/SY8OSZU.png" width="40%"></div>

<div align=center><img src="https://i.imgur.com/fZRB2xh.png" width="40%"></div>

<div align=center><img src="https://i.imgur.com/jXVp3Ns.png" width="80%"></div>

The entire model is based on GCN (Graph Convolutional Neural Network) and RL (reinforcement learning), using GCN to calculate the human state and predict the direction, using reinforcement learning to optimize robot decisions, and combining the two to develop efficient robot navigation in crowds.



### Personal Understanding of RGL Model.

In the process of movement, pedestrians will be affected by other pedestrians. **The graph neural network is used to regard people as nodes and the interaction between pedestrians as edges.** 

The interaction coefficient between pedestrians a and pedestrians b, c and d is calculated through the attention mechanism, and the influence of pedestrians a on pedestrians b, c and d is obtained by weighting, that is, a weighting vector. Use this vector to predict the path of pedestrian a.

This vector is then fused with other features, input into other networks (either fully connected layer or recurrent neural network), and finally trained by supervised learning to obtain the parameters of the graph neural network. This graph neural network refers to the graph composed of human beings, which will have some parameters, this graph neural network is a feature extraction fusion device, the function is to extract the feature vector.

For the prediction of human trajectory, ***LSTM*** can get the position of the next moment with each recursive step, and the trajectory can be obtained by sequential recursion. 

For the robot trajectory prediction part, graph neural network is used, but it is not used to predict the robot position, but is used to form a ***value network*** of ***reinforcement learning***. The node of the graph neural network is the position of the robot and the pedestrian, and the output is the ***value***, which is used to evaluate the quality of the state.



### Ongoing Experiments on Control Variables.

Overall, making the robot visible to pedestrians had a higher success rate, while other factors had little effect.

**Preliminary experimental results:**

<img src="https://i.imgur.com/UiNCPPk.png" width="70%">

<img src="https://i.imgur.com/LmrERuC.png" width="70%">



### Results Visualization:

<table>
    <tr>
    <td><center><img src="https://i.imgur.com/NGmMeST.jpg" width="120%"></center></td>
    <td><center><img src="https://i.imgur.com/ZnBlgn9.jpg" width="120%"></center></td>
    </tr>
    <tr>
    <td><center><img src="https://i.imgur.com/72HBvpJ.jpg" width="120%"></center></td>
    <td><center><img src="https://i.imgur.com/r0Xgaze.jpg" width="120%"></center></td>
    </tr>
</table>

The local fluctuation of model training is large, but the overall trend is normal and tends to converge gradually.
Specific performance:
1. The time to reach the target point is gradually reduced.
2. The success rate of reaching the target point gradually increases.
3. The reward value is increasing each time (indicating that the robot has found the right learning direction).
4. The collision rate is gradually reduced (the success rate of obstacle avoidance is continuously improved).

<table>
    <tr>
    <td><center><img src="https://i.imgur.com/1j6jKR2.gif" width="120%"></center></td>
    <td><center><img src="https://i.imgur.com/jAbMpfV.gif" width="120%"></center></td>
    </tr>
    <tr>
    <td><center><img src="https://i.imgur.com/Ayw2fiD.gif" width="120%"></center></td>
    <td><center><img src="https://i.imgur.com/76eqBHM.gif" width="120%"></center></td>
    </tr>
</table>

*Simulation using gym.*

## Demonstration of Overall System

https://youtu.be/MbLl9JUXypQ

[<img src="https://i.imgur.com/7ToyGK4.png" width="80%">](https://youtu.be/MbLl9JUXypQ)
