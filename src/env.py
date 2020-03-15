import cv2
import numpy as np
import subprocess as sp
from MAMEToolkit.sf_environment import Environment


class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "60", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            # 创建子进程，Shell命令，标准输入，错误句柄
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        # 管道pipe交互操作读／写通信
        self.pipe.stdin.write(image_array.tostring())


# 图像预处理
def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (168, 168))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 168, 168))


# 创建街机游戏模拟环境
class StreetFighterEnv(object):
    def __init__(self, index, monitor = None):
        roms_path = "/home/zhangchao/Downloads/"
        self.env = Environment("env{}".format(index), roms_path)
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None
        self.env.start()

    # 单步执行返回数据：游戏画面，一轮结束，场景切换，游戏结束
    def step(self, action):
        move_action = action//10
        attack_action = action%10
        frames, reward, round_done, stage_done, game_done = self.env.step(move_action, attack_action)

        if self.monitor:
            for frame in frames:
                # 通过管道pipe为子进程写入图像数据
                self.monitor.record(frame)
        if not (round_done or stage_done or game_done):
            # frame拼接
            frames = np.concatenate([process_frame(frame) for frame in frames], 0)[None, :, :, :].astype(np.float32)
        else:
            frames = np.zeros((1, 3, 168, 168), dtype=np.float32)

        # 奖励设计
        reward = reward["P1"]
        if stage_done:
            reward = 25
        elif game_done:
            reward = -50
        reward *= (1+(self.env.stage-1)/10)
        reward /= 10

        return frames, reward, round_done, stage_done, game_done

    # 重启
    def reset(self, round_done, stage_done, game_done):
        if game_done:
            self.env.new_game()
        elif stage_done:
            self.env.next_stage()
        elif round_done:
            self.env.next_round()
        return np.zeros((1, 3, 168, 168), dtype=np.float32)


# 用于不断创建带monitor的训练环境
def create_train_env(index, output_path=None):
    num_inputs = 3
    num_actions = 90

    if output_path:
        monitor = Monitor(384, 224, output_path)
    else:
        monitor = None

    env = StreetFighterEnv(index, monitor)
    return env, num_inputs, num_actions