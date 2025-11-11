import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

from springc_utils import *
import numpy as np
import torch.nn as nn
import torch
import collections
import random
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import os
from configs import config
from webgame_utils.capture_env_state import Env
from webgame_utils.click_actions import *
from springc_utils import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
setup_logging("train.log")

def totensor(r, dtype=torch.float32):
    if isinstance(r, np.ndarray):
        return torch.from_numpy(r).to(device=device, dtype=dtype)
    if isinstance(r, list):
        return torch.from_numpy(np.asarray(r)).to(device=device, dtype=dtype)
    if isinstance(r, torch.Tensor):
        return r.to(device)
    raise ValueError(f"类型不是我想的这三种，当前类型:{type(r)}.")


def tonumpy(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    
class ReplayBuffer():
    
    def __init__(self, state_dim=config.n_actions, max_buffer=config.max_buffer, batch_size=config.batch_size):
        self.state = [None] * max_buffer
        self.next_state = [None] * max_buffer
        self.reward = np.zeros([max_buffer], dtype=np.float32)
        self.action = np.zeros([max_buffer], dtype=np.int32)
        self.done = np.zeros([max_buffer], dtype=np.float32)
        self.max_size = max_buffer
        self.cur_ptr = 0
        self.cur_size = 0
        self.batch_size = batch_size
        
        
    def store(self, s, a, r, s1, done):
        if r == 0:
            a = 1
        self.state[self.cur_ptr] = s
        self.next_state[self.cur_ptr] = s1
        self.reward[self.cur_ptr] = r
        self.action[self.cur_ptr] = a
        self.done[self.cur_ptr] = done
        self.cur_size = min(self.max_size, self.cur_size+1)
        self.cur_ptr = (self.cur_ptr+1)%self.max_size

        
    def sample(self, bs=64):
        idx = np.random.choice(self.cur_size, self.batch_size, replace=False)
        trans = {}
        trans['state'] = self.state[idx]
        trans['next_state'] = self.next_state[idx]
        trans['reward'] = self.reward[idx]
        trans['action'] = self.action[idx]
        trans['done'] = self.done[idx]
        return trans
        
    def size(self):
        return self.cur_size


def autopad(k, p=None):  
    # 对输入的特征层进行自动padding，按照Same原则
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    # 标准卷积+标准化+激活函数
    default_act = nn.SiLU()
    # default_act = nn.ReLU()
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p),  bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DQNet(nn.Module):
    
    def __init__(self):
        super(DQNet, self).__init__()
        self.net = nn.Sequential(
            Conv(3, 128, 7, 4),
            Conv(128, 256, 5, 2),
            Conv(256, 512, 5, 2),
            Conv(512, 512, 3, 1),
            Conv(512, 256, 3, 1),
            )
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.v_head = nn.Linear(256, 1)
        self.a_head = nn.Linear(256, config.n_actions)

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x).flatten(1)
        v = self.v_head(x)
        a = self.a_head(x)
        y = v + a - a.mean(dim=-1, keepdim=True)
        return y

class DQN_Agent():
    
    def __init__(self):
        super().__init__()
        self.minmal_data_size =512
        self.min_eps = 0.1
        self.max_eps = 1.0
        # self.eps = self.max_eps
        self.eps = self.min_eps
        self.eps_decay = 1/1000

        self.main_net = DQNet().to(device)
        self.target_net = DQNet().to(device)
        self.sync_paramters()
        self.replay = ReplayBuffer()
        self.optmizer = torch.optim.Adam(self.main_net.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optmizer, T_max=config.T_max, eta_min=config.min_lr)
        self.freq_update_params = config.freq_update_params
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        self.writer = SummaryWriter(log_dir=f"runs/DQN_example/{TIMESTAMP}")  # TensorBoard writer
        self.env = Env()
        self.pre_img = None
        self.score_max = 0
        self.n_episode = 0


    def preprocess(self, x):
        x = x.astype(np.float32)
        x /= 255.0
        x = np.transpose(x, (2, 0, 1))
        x = totensor(x)[None]
        return x
    
    def sync_paramters(self):
        self.target_net.load_state_dict(self.main_net.state_dict())
        
    def take_action(self, x):
        if np.random.uniform()<self.eps:
            a = np.random.randint(config.n_actions)
        else:
            self.main_net.eval()
            with torch.no_grad():
                a = self.main_net(x)[0].argmax().item()
            self.main_net.train()
        return a
    
    def reset(self):
        self.score = 0
        self.loss = []
        self.trans = []
        self.state, self.next_state = None, None
        self.is_playing = False
        self.env.sd.reset()

    def do_stop(self):
        self.writer.add_scalar("Loss/episode", np.asarray(self.loss).mean(), self.n_episode)
        self.writer.add_scalar("Score/score", self.score, self.n_episode)
        self.reset()


    def train_dqn(self):
        print("START TRAIN...")
        self.main_net.train()
        self.target_net.eval()
        interval = 1 / config.fps

        update_cnt = 0
        update_cnt_total = 0
        self.reset()

        while True:
            start_time = time.time()
            game_img, is_start_page, is_play_page, is_stop_game, score_changed = self.env.capture()
            self.next_state = self.preprocess(game_img)

            if is_start_page:
                time.sleep(0.5)
                click_start_page()

                start_time = time.time()
                game_img = self.env.sct.grab(config.game_region)
                game_img = np.array(game_img)[:, :, :3]
                self.state = self.preprocess(game_img)
                a = self.take_action(self.state)
                click_action(a)
                self.trans.append(self.state)
                self.trans.append(a)
                self.is_playing = True
                elapsed = time.time() - start_time
                if elapsed < interval:
                    time.sleep(interval - elapsed)
                continue


            if is_stop_game: #第一次遇到is_stop_game需要再处理一下
                
                self.is_playing=False
                reward = -10
                self.trans.append(reward)
                self.trans.append(self.next_state)
                self.trans.append(is_stop_game)

                if self.score > self.score_max:
                    self.score_max = self.score
                    torch.save(self.main_net.state_dict(), f"checkpoints\dqn_main_net_best.pth")
                
                # if update_cnt > 4000 and update_cnt%2000==0:#暂停玩游戏，专心训练已有数据
                #     for _  in range(2000):
                #         update_cnt_total += 1
                #         loss = self.update_model()
                self.reset()
                if self.n_episode > 100000:#训练超过10W次完整游戏后，退出
                    break
                else:
                    continue

            if self.is_playing:
                reward = 10 if score_changed else 0.5
                if score_changed:
                    self.score += 1
                self.trans.append(reward)
                self.trans.append(self.next_state)
                self.trans.append(is_stop_game)
                if config.is_debug:
                    print(f"存储数据：{len(self.trans)}")
                self.replay.store(*self.trans)

                self.state = self.next_state
                self.trans = []
                a = self.take_action(self.state)
                click_action(a)
                self.trans.append(self.state)
                self.trans.append(a)

                if self.replay.size() > self.minmal_data_size:
                    update_cnt += 1
                    update_cnt_total += 1
                    self.update_model()
                    if update_cnt % config.freq_update_params == 0:
                        self.sync_paramters()
                    if update_cnt % config.freq_save_model:
                        torch.save(self.main_net.state_dict(), f"checkpoints\dqn_main_net_{update_cnt}.pth")

                elapsed = time.time() - start_time
                if elapsed < interval:
                    time.sleep(interval - elapsed)
                # if config.is_debug:
                print(f"sleep :{interval - elapsed}")

        self.writer.close()
    
    def _compute_dqn_loss(self):
        trans = self.replay.sample() 
        state, next_state, reward, action, done = trans['state'], trans['next_state'], trans['reward'], trans['action'], trans['done']
        state       = torch.cat(state, dim=0)
        next_state  =  torch.cat(next_state, dim=0)
        reward      = totensor(reward, torch.float32)
        action      = totensor(action, torch.long)
        done        = totensor(done, torch.float32)

        target_action = self.main_net(next_state[:, 0], next_state[:, 1]).argmax(dim=1, keepdim=True)
        next_q_value = self.target_net(next_state[:, 0], next_state[:, 1]).gather(1, target_action).detach()
        target = reward + self.gamma * next_q_value[:, 0] * (1-done)
        
        pred = self.main_net(state[:, 0], state[:, 1]).gather(1, action.reshape(-1, 1))
        loss = F.smooth_l1_loss(pred, target.reshape(-1, 1))
        return loss

    def update_model(self):
        loss = self._compute_dqn_loss()
        self.optmizer.zero_grad()
        loss.backward()
        self.optmizer.step()
        return loss.item()
        

    def get_policy(self):
        index = np.asarray([[r, c] for r in range(self.env.n_row) for c in range(self.env.n_row)]).reshape(-1, 2)
        policy = self.main_net(index[:, 0], index[:, 1]).argmax(-1).reshape(self.env.n_row, self.env.n_col)
        return tonumpy(policy)
    
    def save_policy(self, n_episode):
        arr_arrow = self.env.visual_policy(self.get_policy())
        policy_str = "  \n".join(" ".join(row) for row in arr_arrow)
        self.writer.add_text("Policy", policy_str, global_step=n_episode)
    
def main():
    # if config.is_debug:
    checkroot(config.p_mask)
    checkroot(config.p_gamewin)
    agent = DQN_Agent()
    try:
        agent.train_dqn()
    except KeyboardInterrupt:
        print("\n采集停止，退出程序。")

main()