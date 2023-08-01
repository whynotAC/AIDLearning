# 用于加载游戏中的所有资料，包括图片以及音频
from __future__ import print_function
import pygame
import sys
import numpy as np
import random
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def load():
    # 加载各类资源的函数
    # 精灵在不同状态下的图片
    PLAYER_PATH = (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    )

    # 背景图片
    BACKGROUND_PATH = 'assets/sprites/background-black.png'

    # 管道图片所在的地址
    PIPE_PATH = 'assets/sprites/pipe-green.png'

    IMAGES, SOUNDS, HITMASKS = {}, {}, {}

    # 加载成绩数字所需的图片
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # 加载地面的图片
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # 加载声音文件
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'
    
    SOUNDS['die']       = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']       = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']       = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh']       = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']       = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    # 加载背景图
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

    # 加载精灵图
    IMAGES['player'] = (
        pygame.image.load(PLAYER_PATH[0]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[1]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[2]).convert_alpha(),
    )

    # 加载水管
    IMAGES['pipe'] = (
        pygame.transform.rotate(pygame.image.load(PIPE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH).convert_alpha(),
    )

    # 获得水管的蒙版
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # 玩家的蒙版
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    # 返回了三个字典，每个字典的值分别存储图像、声音和蒙版
    return IMAGES, SOUNDS, HITMASKS

def getHitmask(image):
    """根据图像的alpha，获得蒙版"""
    # 所谓蒙版就是将图像中的主体从整个图像中抠出来的技术，从而方便与其它的对象合成到一起
    # 蒙版用一个boolean类型的列表来存储
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

# 实现Flappy Bird的游戏逻辑
FPS = 30 # 帧率
SCREENWIDTH = 288
SCREENHEIGHT = 512

pygame.init()           # 游戏初始化
FPSCLOCK = pygame.time.Clock()      # 定义程序时钟
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))       # 定义屏幕对象
pygame.display.set_caption('Flappy Bird')       # 设置窗口名称

IMAGES, SOUNDS, HITMASKS = load()
PIPEGAPSIZE = 100           # 定义两个水管之间的宽度
BASEY = SCREENHEIGHT * 0.79 # 设置基地的高度

# 设定小鸟属性: 宽度、高度
PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()

# 设定水管属性: 高度、宽度
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()

# 背景宽度
BACKGROUND_WIDTH = IMAGES['background'].get_width()

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

# 游戏模型类
class GameState:
    def __init__(self):
        # 初始化
        self.score = self.playerIndex = self.loopIter = 0

        # 设定玩家的初始位置
        self.playerx = int(SCREENWIDTH*0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0

        # 地面的初始位移
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH

        # 生成两个随机的水管
        newPipe1 = getRandomPipe()
        newPipe2 = getRandomPipe()

        # 设定初始水管的位置x,y坐标
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        # 定义玩家的属性
        self.pipeVelX = -4
        self.playerVelY = 0             # 小鸟在y轴上的速度
        self.playerMaxVelY = 10         # y轴上的最大速度
        self.playerMinVelY = -8         # y轴向上的最大速度
        self.playerAccY    = 1          # 小鸟往下落的加速度
        self.playerFlapAcc = -9         # 煽动翅膀的加速度
        self.playerFlapped = False      # 玩家是否煽动了翅膀
    
    def frame_step(self, input_actions):
        # input_actions是一个行动数组，分别存储了0或者1两个动作的激活情况
        # 游戏每一帧的循环
        pygame.event.pump()

        # 每一步的默认回报
        reward = 0.1
        terminal = False

        # 限定每一帧只能做一个动作
        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: 对应什么都不做
        # input_actions[1] == 1: 对应小鸟煽动了翅膀
        if input_actions[1] == 1:
            # 小鸟煽动翅膀向上
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                SOUNDS['wing'].play()

        # 检查是否通过了管道，通过，则增加成绩
        playerMidPos = self.playerx + PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                SOUNDS['point'].play()
                reward = 1
        
        # playerIndex轮换
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # 小鸟运动
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0
        
        # 管道的移动
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX
        
        # 当管道快到左侧边缘的时候，产生新的管道
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])
        
        # 当第一个管道溢出屏幕的时候，就将其删除
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)
        
        # 检查碰撞
        isCrash = checkCrash({'x': self.playerx, 'y': self.playery, 'index': self.playerIndex}, self.upperPipes, self.lowerPipes)

        # 如果有碰撞发生，则游戏结束
        if isCrash:
            SOUNDS['hit'].play()
            SOUNDS['die'].play()
            terminal = True
            self.__init__()
            reward = -1
        
        # 将所有角色都根据每个角色的坐标画在屏幕上
        SCREEN.blit(IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        showScore(self.score)
        SCREEN.blit(IMAGES['player'][self.playerIndex], (self.playerx,   self.playery))

        # 将当前的游戏屏幕生成一个二维画面返回
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        return image_data, reward, terminal

def getRandomPipe():
    # 随机生成管道的函数
    # 两个管道之间的竖直间隔从下列数中直接取
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
    index = random.randint(0, len(gapYs)-1)
    gapY = gapYs[index]

    # 设定新生成管道的位置
    gapY += int(BASEY * 0.2)
    pipeX = SCREENWIDTH + 10

    # 返回管道的坐标
    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},      # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},      # lower pipe
    ]

def showScore(score):
    # 在屏幕上直接展示成绩的函数
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0      # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()
    
    Xoffset = (SCREENWIDTH - totalWidth) / 2
    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()
    
def checkCrash(player, upperPipes, lowerPipes):
    # 检查碰撞的函数，基本思路为: 将每一个物体都看做是一个矩形区域，然后检查两个矩形区域是否有碰撞
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][1].get_height()

    # 检查小鸟是否碰撞到地面
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    else:
        # 检查小鸟是否与管道碰撞
        playerRect = pygame.Rect(player['x'], player['y'], player['w'], player['h'])

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # 上下管道矩形
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            # 获取每个元素的蒙版
            pHitMask = HITMASKS['player'][pi]
            uHitMask = HITMASKS['pipe'][0]
            lHitMask = HITMASKS['pipe'][1]

            # 检查是否与上下管道相撞
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitMask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitMask)

            if uCollide or lCollide:
                return True
    return False

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    # 在像素级别检查两个物体是否发生碰撞
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False
    
    # 确定矩形框
    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

# 新建一个游戏---- 游戏测试
# game = GameState()

# fig = plt.figure()
# axe = fig.add_subplot(111)
# dat = np.zeros((10,10))
# img = axe.imshow(dat)

# # 进行100步循环
# for i in range(100):
#     clear_output(wait=True)
#     image_data, reward, terminal = game.frame_step([0,1])

#     image = np.transpose(image_data, (1, 0, 2))
#     img.set_data(image)
#     img.autoscale()
#     display(fig)

# 导入必须的包
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import sys
sys.path.append("game/")
import random
import numpy as np
from collections import deque

# 模型定义
GAME = 'bird' # 游戏名称
ACTIONS = 2 # 有效输出动作的个数
GAMMA = 0.99 # 强化学习中未来的衰减率
OBSERVE = 10000. # 训练之前的时间步，需要先观察10000帧
EXPLORE = 3000000. # 退火所需的时间步，所谓的退火就是指随机选择率epsilon逐渐变小
FINAL_EPSILON = 0.0001 # epsilon的最终值
INITIAL_EPSILON = 0.1 # epsilon的初始值
REPLAY_MEMORY = 50000 # 最多记忆多少帧训练数据
BATCH = 32 # 每一个批次的数据记录条数
FRAME_PER_ACTION = 1 # 每间隔多少时间完成一次有效动作的输出

# 创建一个多层CNN网络，该网络接收的输入为4帧画面，输出为每个可能动作对应的Q函数值
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一层卷积，从4通道到32通道，窗口大小8，跳跃间隔4，填空白2
        self.conv1 = nn.Conv2d(4, 32, 8, 4, padding=2)
        # Pooling层，窗口2*2
        self.pool = nn.MaxPool2d(2, 2)
        # 第二层卷积，从32通道到64通道，窗口大小4，跳跃间隔2，填空白1
        self.conv2 = nn.Conv2d(32, 64, 4, 2, padding=1)
        # 第二个Pooling层，窗口2*2，空白1
        #self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        # 第三层卷积层，输入输出通道都是64，填空白为1
        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding=1)

        # 最后有两层全连接层
        self.fc_sz = 1600
        self.fc1 = nn.Linear(self.fc_sz, 256)
        self.fc2 = nn.Linear(256, ACTIONS)
    
    def forward(self, x):
        # 输入为一个batch的数据
        # x的尺寸为: batch_size, 4, 80, 80
        x = self.conv1(x)
        # x的尺寸为: batch_size, 32, 20, 20
        x = F.relu(x)
        x = self.pool(x)
        # x的尺寸为: batch_size, 32, 10, 10
        x = F.relu(self.conv2(x))
        # x的尺寸为：batch_size, 64, 5, 5
        #x = self.pool2(x)
        x = F.relu(self.conv3(x))
        # x的尺寸为：batch_size, 64, 5, 5
        #x = self.pool2(x)
        # 将x设为1600维的向量, batch_size, 1600
        x = x.view(-1, self.fc_sz)
        x = F.relu(self.fc1(x))
        readout = self.fc2(x)
        return readout, x

    def init(self):
        # 初始化所有的网络权重
        self.conv1.weight.data = torch.abs(0.01 * torch.randn(self.conv1.weight.size()))
        self.conv2.weight.data = torch.abs(0.01 * torch.randn(self.conv2.weight.size()))
        self.conv3.weight.data = torch.abs(0.01 * torch.randn(self.conv3.weight.size()))
        self.fc1.weight.data = torch.abs(0.01 * torch.randn(self.fc1.weight.size()))
        self.fc2.weight.data = torch.abs(0.01 * torch.randn(self.fc2.weight.size()))
        self.conv1.bias.data = torch.ones(self.conv1.bias.size()) * 0.01
        self.conv2.bias.data = torch.ones(self.conv2.bias.size()) * 0.01
        self.conv3.bias.data = torch.ones(self.conv3.bias.size()) * 0.01
        self.fc1.bias.data = torch.ones(self.fc1.bias.size()) * 0.01
        self.fc2.bias.data = torch.ones(self.fc2.bias.size()) * 0.01

use_cuda = torch.cuda.is_available()
# 初始化一个神经网络
net = Net()
net.init()
net = net.cuda() if use_cuda else net

# 定义损失函数为MSE
criterion = nn.MSELoss().cuda() if use_cuda else nn.MSELoss()
# 定义优化器，并设置初始学习率维10^-6
optimizer = torch.optim.Adam(net.parameters(), lr=1e-6)

# 开启一个游戏进程，开始与游戏引擎通话
game_state = GameState()

# 学习样本的存储区域deque是一个类似于list的存储容器
D = deque()

# 状态打印Log记录位置
# a_file = open("logs_" + GAME + "/readout.txt", 'w')
# h_file = open("logs_" + GAME + "/hidden.txt", 'w')

# 将游戏设置为初始状态，并获得一个80*80的游戏界面
do_nothing = np.zeros(ACTIONS)
do_nothing[0] = 1
x_t, r_0, terminal = game_state.frame_step(do_nothing)
x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)

# 将初始的游戏画面叠加成4张作为神经网络的初始输入状态s_t
s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

# 设置初始的epsilon（采取随机行动的概率）,并准备训练
epsilon = INITIAL_EPSILON
t = 0

# 记录每轮平均得分的容器
scores = []
all_turn_scores = []
while "flappy bird" != "angry bird":
    # 开始游戏循环
    s = torch.from_numpy(s_t).type(torch.FloatTensor)
    s = s.cuda() if use_cuda else s
    s = s.view(-1, s.size()[0], s.size()[1], s.size()[2])
    # 获取当前时刻的游戏画面
    readout, h_fc1 = net(s)
    # 神经网络产生的输出为readout
    readout = readout.cpu() if use_cuda else readout
    # readout为一个二维向量，分别对应每一个动作的预期Q值
    readout_t = readout.data.numpy()[0]

    # 按照epsilon贪婪策略产生小鸟的行动
    a_t = np.zeros([ACTIONS])
    action_index = 0
    if t % FRAME_PER_ACTION == 0:
        if random.random() <= epsilon:
            # print("-------------Random Action-------------")
            action_index = random.randrange(ACTIONS)
        else:
            action_index = np.argmax(readout_t)
        a_t[action_index] = 1
    else:
        a_t[0] = 1 # do nothing
    
    if epsilon > FINAL_EPSILON and t > OBSERVE:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
    
    # 其次，将选择好的行动输入给游戏引擎，并得到下一帧的状态
    x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

    scores.append(r_t)
    if terminal:
        all_turn_scores.append(sum(scores))
        scores= []
    
    x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1, (1, 80, 80))
    # 将当前帧的画面和前三帧的画面合并起来作为Agent获得的环境反馈结果
    s_t1 = np.append(x_t1, s_t[:3, :, :], axis=0)
    # 生成一个训练数据
    D.append((s_t, a_t, r_t, s_t1, terminal))
    if len(D) > REPLAY_MEMORY:
        D.popleft()
    
    # 最后，当运行周期超过一定次数后开始训练神经网络
    if t > OBSERVE:
        # 从D中随机采样出一个batch的训练数据
        minibatch = random.sample(D, BATCH)
        optimizer.zero_grad()

        # 将这个batch中的s变量都分别存放在列表中
        s_j_batch = [d[0] for d in minibatch]
        a_batch = [d[1] for d in minibatch]
        r_batch = [d[2] for d in minibatch]
        s_j1_batch = [d[3] for d in minibatch]

        # 接下来，根据s_j1_batch,神经网络给出预估的未来Q值
        s = torch.tensor(np.array(s_j1_batch, dtype=float), dtype=torch.float)
        s = s.cuda() if use_cuda else s
        readout, h_fc1 = net(s)
        readout = readout.cpu() if use_cuda else readout
        readout_j1_batch = readout.data.numpy()
        # readout_j1_batch存储了一个Minibatch中的所有未来一步的Q预估值
        # 根据Q的预估值，当前的反馈r，以及游戏是否结束，更新带训练的目标函数
        y_batch = []
        for i in range(0, len(minibatch)):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(r_batch[i])
            else:
                y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
        
        # 开始梯度更新
        y = torch.tensor(y_batch, dtype = torch.float, requires_grad=True)
        a = torch.tensor(a_batch, dtype = torch.float, requires_grad=True)
        s = torch.tensor(np.array(s_j_batch, dtype=float), dtype=torch.float, requires_grad=True)
        if use_cuda:
            y = y.cuda()
            a = a.cuda()
            s = s.cuda()
        # 计算s_j_batch的Q值
        readout, h_fc1 = net(s)
        readout_action = readout.mul(a).sum(1)
        # 根据s_j_batch下所选择的预估Q和目标y的Q值的差来作为损失函数训练网络
        loss = criterion(readout_action, y)
        loss.backward()
        optimizer.step()
        if t % 1000 == 0:
            print('损失函数: ', loss)
    
    # 将状态更新一次,时间步+1
    s_t = s_t1
    t += 1

    if t % 10000 == 0:
        torch.save(net, 'saving_nets/' + GAME + '-dqn' + str(t) + '.txt')
    
    # 状态信息的转化，分别为Observe,explore和train三个阶段
    state = ""
    if t <= OBSERVE:
        state = "observe"
    elif t > OBSERVE and t <= OBSERVE + EXPLORE:
        state = "explore"
    else:
        state = "train"
    
    # 打印当前运行的一些基本数据，分别输出到屏幕以及log文件中
    if t % 1000 == 0:
        sss = "时间步 {}/ 状态 {}/ Epsilon {:.2f}/ 行动 {}/ 奖励 {}/ Q_MAX {:e}/ 轮得分 {:.2f}".format(
            t, state, epsilon, action_index, r_t, np.max(readout_t), np.mean(all_turn_scores[-1000:]))
        print(sss)
        f = open('log_file.txt', 'a')
        f.write(sss + '\n')
        f.close()

f = open('final_log_file.txt', 'r')
line = f.read().strip().split('\n')
values = []
for ln in line:
    segs = ln.split('/')
    values.append(float(segs[-1].split(' ')[-1]))
plt.figure()
plt.plot(np.arange(len(values))*1000, values)
plt.xlabel('Frames')
plt.ylabel('Average Score')
plt.show()


""" 已经训练好的模型
net = torch.load('final_model.mdl')
FINAL_EPSILON = 0.0001      # epsilon的最终值
BATCH = 32                  # 每一个批次的数据记录条数
FRAME_PER_ACTION = 1        # 每间隔多少时间完成一次有效动作的输出

use_cuda = torch.cuda.is_available()

net = net.cuda() if use_cuda else net

game_state = GameState()

# 将游戏设置为初始状态，并获得一个80*80的游戏页面
do_nothing = np.zeros(ACTIONS)
do_nothing[0] = 1
x_t, r_0, terminal = game_state.frame_step(do_nothing)
x_t = cv2.cvtColor(cv2.resize(x_t, (80,80)), cv2.COLOR_BGR2GRAY)
ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)

# 将初始化的游戏画面跌成4张作为神经网络的初始输入状态s_t
s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

# 设置初始的epsilon，并准备训练
epsilon = FINAL_EPSILON
t = 0       # 记录每轮平均得分的容器
scores = []
all_turn_scores = []

fig = plt.figure()
axe = fig.add_subplot(111)
dat = np.zeros((10, 10))
img = axe.imshow(dat)
while "flappy bird" != "angry bird":
    # 开始游戏循环
    # 首先，按照贪婪策略选择一个行动
    s = torch.from_numpy(s_t).type(torch.FloatTensor).requires_grad_(False)
    s = s.cuda() if use_cuda else s
    s = s.view(-1, s.size()[0], s.size()[1], s.size()[2])
    # 获取当前时刻的游戏画面，输入到神经网络
    readout, h_fc1 = net(s)
    # 神经网络产生的输出为readout: 选择每一个行动的预期Q值
    readout = readout.cpu() if use_cuda else readout
    # readout为一个二维向量，分别对应每一个动作的预期Q值
    readout_t = readout.data.numpy()[0]

    # 按照epsilon贪婪策略产生小鸟的行动，即以epsilon的概率随机输出行动或者以1-epsilon的概率
    a_t = np.zeros([ACTIONS])
    action_index = 0
    if t % FRAME_PER_ACTION == 0:
        # 如果当前帧可以行动，则
        if random.random() <= epsilon:
            # 产生随机行动
            action_index = random.randrange(ACTIONS)
        else:
            # 选择神经网络判断的预期Q最大的行动
            action_index = np.argmax(readout_t)
        a_t[action_index] = 1
    else:
        a_t[0] = 1  # do nothing
    # 其次，将选择好的行动输入给游戏引擎，并得到下一帧的状态
    x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
    # 返回的x_t1_colored为游戏画面，r_t为本轮得分，terminal为游戏在本轮是否已经结束

    scores.append(r_t)
    if terminal:
        all_turn_scores.append(sum(scores))
        scores = []
    
    # 对游戏的初始画面作相应的处理，从而变成80*80的
    x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1, (1, 80, 80))
    # 将当前帧的画面和前三帧的画面合并起来作为Agent获得的环境反馈结果
    s_t1 = np.append(x_t1, s_t[:3, :, :], axis=0)
    s_t = s_t1
    t += 1
    clear_output(wait=True)

    image = np.transpose(x_t1_colored, (1, 0, 2))
    img.set_data(image)
    img.autoscale()
    display(fig)
"""
