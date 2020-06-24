import os
import sys
import signal
import argparse
import math
import threading
import torch
import numpy as np
from copy import deepcopy
from lib.rl.ddpg import DDPG
from tensorboardX import SummaryWriter

sys.path.append(os.path.join(os.getcwd(), "comp_reram"))

from comp_reram import *

# feature: (is_conv, in_channel, out_channel, filter_size, weight_size, in_feature, layer_idx, bits)
# lenet_info = [[1,   1,   6, 5,    6*1*5*5, 1*32*32, 1, 6, 6, 6, 6], \
#               [1,   6,  16, 5,   16*6*5*5, 6*14*14, 2, 6, 6, 6, 6], \
#               [1,  16, 120, 5, 120*16*5*5,  16*5*5, 3, 6, 6, 6, 6], \
#               [0, 120,  10, 1, 10*120*1*1, 120*1*1, 4, 6, 6, 6, 6]]

vgg16_info = \
[[1,      3,    64, 3,      3*64*3*3,   3*224*224,  1 ], \
 [1,     64,    64, 3,     64*64*3*3,  64*224*224,  2 ], \
 [1,     64,   128, 3,    64*128*3*3,  64*112*112,  3 ], \
 [1,    128,   128, 3,   128*128*3*3, 128*112*112,  4 ], \
 [1,    128,   256, 3,   128*256*3*3,   128*56*56,  5 ], \
 [1,    256,   256, 3,   256*256*3*3,   256*56*56,  6 ], \
 [1,    256,   256, 3,   256*256*3*3,   256*56*56,  7 ], \
 [1,    256,   512, 3,   256*512*3*3,   256*28*28,  8 ], \
 [1,    512,   512, 3,   512*512*3*3,   512*28*28,  9 ], \
 [1,    512,   512, 3,   512*512*3*3,   512*28*28, 10 ], \
 [1,    512,   512, 3,   512*512*3*3,   512*14*14, 11 ], \
 [1,    512,   512, 3,   512*512*3*3,   512*14*14, 12 ], \
 [1,    512,   512, 3,   512*512*3*3,   512*14*14, 13 ]]#, \
 # [0,512*7*7,  4096, 1,  512*7*7*4096,     512*7*7, 14 ], \
 # [0,   4096,  4096, 1,     4096*4096,        4096, 15 ], \
 # [0,   4096,  1000, 1,     4096*1000,        4096, 16 ]]

# tunable_params = [4, 8, 4, 8, 9, 4, 8]
tunable_params = [16, 8, 16, 8, 9]

vgg16_info = [ lst+tunable_params for lst in vgg16_info ]


def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))


class QuantEnv:
    def __init__(self, model_info, batch_size=16):
        self.quant_scheme = []  # quantization strategy
        self.layer_feature = self.normalize_feature(model_info)
        self.wsize_list = [6*1*5*5, 16*6*5*5, 120*16*5*5, 10*120*1*1]
        self.cur_ind = 0
        # self.bound_list = [(2,5), (2,5), (2,5), (2,5), (4,9), (2,5), (2,5)]
        self.bound_list = [(2,5), (2,5), (2,5), (2,5), (4,9)]
        self.total_bitwidth_list = [8, 16, 32]
        self.adc_bitwidth_list = [4,5,6,7,8,9]
        self.last_action = [(16,8,16,8,9)]
        self.org_acc = 0.9837
        self.best_reward = -math.inf
        self.original_wsize = sum([ e*16 for e in self.wsize_list])

        self.bw_weights = self.norm_bw(vgg16_info)

        self.QE = QuantEvaluator(batch_size=batch_size)

        self.global_index = 0

    def reset(self):
        self.cur_ind = 0
        self.quant_scheme = []
        obs = self.layer_feature[0].copy()
        return obs

    def cost_estimate(self, quant_scheme=None):
        tmp = [ [(qs[0]+qs[1])/13, (qs[2]+qs[3])/13] for qs in quant_scheme ]
        adc_tmp = float(sum([ qs[4] for qs in quant_scheme ])) / 13

        qs_p = np.array(tmp, 'float')
        assert(qs_p.shape == self.bw_weights.shape)
        cost = qs_p * self.bw_weights * (1.25 ** adc_tmp)
        return sum(sum(cost))

    def norm_bw(self, info):
        model_info = np.array(info, 'float')[:, 3:5]
        sum_quant = sum(sum(model_info))
        return model_info / sum_quant

    def reward(self, loss_diff, quant_scheme):
        return -(loss_diff/10 + self.cost_estimate(quant_scheme))

    def step(self, action):
        action = self._action_wall(action)
        self.quant_scheme.append(action)

        # all the actions are made
        if self.cur_ind == 12:

            self.global_index += 1

            log_file = open("./logs/"+str(self.global_index)+".txt", "a+")

            # self._final_action_wall()
            assert len(self.quant_scheme) == len(self.layer_feature)

            q_scheme = [ [2, 1]+(qs)+[16, 12] for qs in self.quant_scheme ]

            # for e in q_scheme:
            #     for i in [2, 3, 4, 5, 7, 8]:
            #         e[i] = 2 * ((e[i]+1)//2)

            # for e in q_scheme:
            #     for i in [2, 4, 7]:
            #         e[i] = e[i] + e[i+1]

            # acc = 0 # TODO
            loss, loss_ref = self.QE.evaluate(q_scheme)
            loss_diff = loss - loss_ref
            print("quant_scheme = ", self.quant_scheme)
            print("loss = ", loss)
            print("loss_diff = ", loss_diff)


            reward = self.reward(loss_diff, self.quant_scheme)
            print("reward = ", reward)

            log_file.write(str(self.quant_scheme) + "\n")
            log_file.write(str(loss) + "\n")
            log_file.write(str(loss_ref) + "\n")
            log_file.write(str(loss_diff) + "\n")
            log_file.write(str(reward) + "\n")

            w_size = 0# sum([ self.quant_scheme[i]*self.wsize_list[i] for i in range(len(self.quant_scheme)) ])
            w_size_ratio = 0# float(w_size) / float(self.original_wsize)

            info_set = {'w_ratio': w_size_ratio, 'loss': loss, 'loss_diff': loss_diff, 'w_size': w_size}

            if reward > self.best_reward:
                self.best_reward = reward
                prGreen('New best policy: {}, reward: {:.3f}, loss: {:.3f}, loss_diff: {:.3f}, w_ratio: {:.3f}'.format(
                    self.quant_scheme, self.best_reward, loss, loss_diff, w_size_ratio))

            obs = self.layer_feature[self.cur_ind, :].copy()  # actually the same as the last state
            done = True


            log_file.close()
            return obs, reward, done, info_set

        w_size = 0 #sum([ self.quant_scheme[i]*self.wsize_list[i] for i in range(len(self.quant_scheme)) ])
        info_set = {'w_size': w_size}
        reward = 0
        done = False
        self.cur_ind += 1  # the index of next layer
        self.layer_feature[self.cur_ind][-5:] = action
        # build next state (in-place modify)
        obs = self.layer_feature[self.cur_ind, :].copy()
        return obs, reward, done, info_set

    def _action_wall(self, actions):
        assert len(self.quant_scheme) == self.cur_ind

        # print("_action_wall BEFORE: ", actions)

        converted_actions = []
        # limit the action to certain range
        for idx, action in enumerate(actions):
            action = float(action)
            if idx in [0, 2]: 
                index = math.floor(action * len(self.total_bitwidth_list))
                if index >= len(self.total_bitwidth_list):
                    index = len(self.total_bitwidth_list) - 1
                converted_actions += [self.total_bitwidth_list[index]]
            elif idx in [1, 3]: 
                converted_actions += [ math.floor(action * converted_actions[idx - 1]) ]
            elif idx == 4:
                index = math.floor(action * len(self.adc_bitwidth_list))
                if index >= len(self.adc_bitwidth_list):
                    index = len(self.adc_bitwidth_list) - 1
                converted_actions += [self.adc_bitwidth_list[index]]

        self.last_action = converted_actions

        # print("_action_wall AFTER: ", converted_actions)
        return converted_actions  # not constrained here

    def normalize_feature(self, model_info):
        # normalize the state
        model_info = np.array(model_info, 'float')
        print('=> shape of feature (n_layer * n_dim): {}'.format(model_info.shape))
        assert len(model_info.shape) == 2, model_info.shape
        # print(f"model_info.shape[1] = {model_info.shape[1]}")
        for i in range(model_info.shape[1]):
            fmin = min(model_info[:, i])
            fmax = max(model_info[:, i])
            if fmax - fmin > 0:
                model_info[:, i] = (model_info[:, i] - fmin) / (fmax - fmin)

        return model_info


def load_state(checkpoint_dir, agent):
    agent.load_weights(checkpoint_dir)
    train_state_file = os.path.join(checkpoint_dir, 'train_state.pkl')

    train_state = {}
    if os.path.exists(train_state_file):
        train_state = torch.load(train_state_file)

    if not isinstance(train_state, dict):
        print("[ERROR]: Train state file does not store python dictionary (type = {})".format(type(train_state)))
        train_state = {}
    return train_state


def save_state(checkpoint_dir, agent, **kwargs):
    agent.save_model(checkpoint_dir)
    torch.save(kwargs, os.path.join(checkpoint_dir, 'train_state.pkl'))


def train(num_episode, agent, env, **kwargs):
    """
    Args:
        num_episode (int): Number of episodes to play.
        agent (DDPG): RL agent.
        env (QuantEnv): RL environment
        kwargs (dict): Additional arguments
    """
    #
    stop_event = threading.Event()
    def _signal_handler(signum, frame):  # noqa e306
        stop_event.set()
    signal.signal(signal.SIGUSR1, _signal_handler)

    output = kwargs.get('output', './save')
    debug = kwargs.get('debug', False)
    warmup = kwargs.get('warmup', 20)
    n_update = kwargs.get('n_update', 20)

    tfwriter = SummaryWriter(logdir=output)
    text_writer = open(os.path.join(output, 'log.txt'), 'w')

    # Variables to serialize (+ agent)
    train_state = load_state(output, agent)
    if len(train_state) != 0:
        print(f"[INFO] Restoring train state from path: {output} ({train_state})")
    best_reward = train_state.get('best_reward', -math.inf)
    best_policy = train_state.get('best_policy', [])
    step = train_state.get('step', 0)
    episode = train_state.get('episode', 0)

    # Variables that do not need to be serialized
    observation = None
    episode_steps = 0
    episode_reward = 0.
    trajectory = []

    print(f"episode = {episode}")
    agent.is_training = True
    while episode < num_episode:  # counting based on episode
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if episode <= warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation, episode=episode)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)

        trajectory.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

        # [optional] save intermediate model
        if stop_event.is_set() or episode % int(num_episode / 10) == 0:
            save_state(output, agent, best_reward=best_reward, best_policy=best_policy, step=step, episode=episode)

        if stop_event.is_set():
            print("[WARNING] Stop event is set, exiting ...")
            break

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode
            msg = '#{}: episode_reward:{:.4f} acc: {:.4f}, weight: {:.4f} MB'.format(
                episode, episode_reward, info['loss'], info['w_ratio'] * 1. / 8e6)
            text_writer.write(msg + '\n')
            if debug:
                print(msg)

            final_reward = trajectory[-1][0]
            # agent observe and update policy
            for i, (r_t, s_t, s_t1, a_t, done) in enumerate(trajectory):
                agent.observe(final_reward, s_t, s_t1, a_t, done)
                if episode > warmup:
                    for _ in range(n_update):
                        agent.update_policy()

            agent.memory.append(observation, agent.select_action(observation, episode=episode), 0., False)

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            trajectory = []

            if final_reward > best_reward:
                best_reward = final_reward
                best_policy = env.quant_scheme

            tfwriter.add_scalar('reward/last', final_reward, episode)
            tfwriter.add_scalar('reward/best', best_reward, episode)
            tfwriter.add_scalar('info/loss', info['loss'], episode)
            tfwriter.add_scalar('info/loss_diff', info['loss_diff'], episode)
            tfwriter.add_scalar('info/w_ratio', info['w_ratio'], episode)
            tfwriter.add_text('info/best_policy', str(best_policy), episode)
            tfwriter.add_text('info/current_policy', str(env.quant_scheme), episode)
            tfwriter.add_scalar('value_loss', agent.get_value_loss(), episode)
            tfwriter.add_scalar('policy_loss', agent.get_policy_loss(), episode)
            tfwriter.add_scalar('delta', agent.get_delta(), episode)
            # record the preserve rate for each layer
            # for i, preserve_rate in enumerate(env.quant_scheme):
            #     tfwriter.add_scalar('preserve_rate_w/{}'.format(i), preserve_rate, episode)

            text_writer.write(f"best reward: {best_reward}\n")
            text_writer.write(f"best policy: {best_policy}\n")

            print(f"episode = {episode}")

    text_writer.close()
    return best_policy, best_reward


def print_stop_me_message():
    print("--------------------------------------------------------------")
    print("ReRAM Search process is running, process ID = {}.".format(os.getpid()))
    print("Send me a USR1 signal to request stop :")
    print("    kill -USR1 {}".format(os.getpid()))
    print("I will terminate myself as soon as I complete playing current episode.")
    print("--------------------------------------------------------------")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description='PyTorch Reinforcement Learning')

    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')

    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 1)')
    # env
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to use)')
    parser.add_argument('--dataset_root', default='data/imagenet', type=str, help='path to dataset)')
    parser.add_argument('--preserve_ratio', default=0.1, type=float, help='preserve ratio of the model size')
    parser.add_argument('--min_bit', default=3, type=float, help='minimum bit to use')
    parser.add_argument('--max_bit', default=6, type=float, help='maximum bit to use')
    parser.add_argument('--float_bit', default=32, type=int, help='the bit of full precision float')
    parser.add_argument('--is_pruned', dest='is_pruned', action='store_true')
    # ddpg
    parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for actor')
    parser.add_argument('--lr_a', default=1e-4, type=float, help='learning rate for actor')
    parser.add_argument('--warmup', default=20, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=1., type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=128, type=int, help='memory size for each layer')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    # noise (truncated normal distribution)
    parser.add_argument('--init_delta', default=0.5, type=float,
                        help='initial variance of truncated normal distribution')
    parser.add_argument('--delta_decay', default=0.99, type=float,
                        help='delta decay during exploration')
    parser.add_argument('--n_update', default=1, type=int, help='number of rl to update each time')
    # training
    parser.add_argument('--max_episode_length', default=1e9, type=int, help='')
    parser.add_argument('--output', default='./save', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_episode', default=600, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=234, type=int, help='')
    parser.add_argument('--n_worker', default=32, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=256, type=int, help='number of data batch size')
    parser.add_argument('--finetune_epoch', default=1, type=int, help='')
    parser.add_argument('--finetune_gamma', default=0.8, type=float, help='finetune gamma')
    parser.add_argument('--finetune_lr', default=0.001, type=float, help='finetune gamma')
    parser.add_argument('--finetune_flag', default=True, type=bool, help='whether to finetune')
    parser.add_argument('--use_top5', default=False, type=bool, help='whether to use top5 acc in reward')
    parser.add_argument('--train_size', default=20000, type=int, help='number of train data size')
    parser.add_argument('--val_size', default=10000, type=int, help='number of val data size')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # Architecture
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='mobilenet_v2', choices=model_names,
    #                 help='model architecture:' + ' | '.join(model_names) + ' (default: mobilenet_v2)')
    # device options
    parser.add_argument('--gpu_id', default='3', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    # Create output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    if not os.path.isdir(args.output):
        raise ValueError("Output directory is not a directory: {}".format(args.output))

    print("Creating QuantEnv environment ...")
    env = QuantEnv(vgg16_info, batch_size=args.batch_size)

    print("Creating DDPG agent ...")
    nb_states = env.layer_feature.shape[1]
    nb_actions = len(tunable_params)
    agent = DDPG(nb_states, nb_actions, args)

    print("Start training ...")
    print_stop_me_message()
    best_policy, best_reward = train(args.train_episode, agent, env, output=args.output, debug=args.debug,
                                     warmup=args.warmup, n_update=args.n_update)

    print(f"Best policy: {best_policy}, best reward: {best_reward}")


if __name__ == '__main__':
    main()
