import os, sys
import torch
import argparse
import numpy as np
import math
from copy import deepcopy
from tensorboardX import SummaryWriter

sys.path.insert(0, './mnist/')
sys.path.insert(0, './lib/rl/')
from evaluate_quant import *
from ddpg import DDPG

# feature: (is_conv, in_channel, out_channel, filter_size, weight_size, in_feature, layer_idx, bits)
lenet_info = [[1,   1,   6, 5,    6*1*5*5, 1*32*32, 1, 6, 6, 6, 6], \
              [1,   6,  16, 5,   16*6*5*5, 6*14*14, 2, 6, 6, 6, 6], \
              [1,  16, 120, 5, 120*16*5*5,  16*5*5, 3, 6, 6, 6, 6], \
              [0, 120,  10, 1, 10*120*1*1, 120*1*1, 4, 6, 6, 6, 6]]

lenet_flops = [32*32*1*5*5*6, 14*14*6*5*5*16, 5*5*16*5*5*120, 10*120]
lenet_size = [(data[5]*12,data[4]*12) for data in lenet_info]

def costFn(acc, quant):
    """
        Get the cost.
    """
    FULL_PREC = 12.0
    FULL_ACC = 0.9837

    acc_diff = FULL_ACC - acc

    act_sizes = [sum(q[:2]) for q in quant]
    wgt_sizes = [sum(q[2:]) for q in quant]

    weight_ratio = sum([act_size[i]*lenet_size[i][0] + wgt_size[i]*lenet_size[i][1] for i in range(4)])/float(FULL_PREC * sum(lenet_size))
    flops_ratio = sum([(max(act_size[i],wgt_size[i])/FULL_PREC) *lenet_flops[i] for i in range(4)])/float(sum(lenet_flops))

    reward = acc_diff + 1.0/weight_ratio + 1.0/flops_ratio

    return reward


def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))

class QuantEnv:
    def __init__(self, model_info):
        self.quant_scheme = [] # quantization strategy
        self.layer_feature = self.normalize_feature(model_info)
        self.wsize_list = [6*1*5*5, 16*6*5*5, 120*16*5*5, 10*120*1*1]
        self.cur_ind = 0
        self.bound_list = [(3,6), (3,6), (3,6), (3,6)]
        self.last_action = [(6,6,6,6)]
        self.org_acc = 0.9837
        self.best_reward = -np.inf
        self.original_wsize = sum([ e*16 for e in self.wsize_list])

    def reset(self):
        self.cur_ind = 0
        self.quant_scheme = []
        obs = self.layer_feature[0].copy()
        return obs

    # for quantization
    def reward(self, acc, w_size_ratio=None, quant_scheme=None):
        if w_size_ratio is not None:
            return (acc - self.org_acc + 1. / w_size_ratio) * 0.1
        if quant_scheme is not None:
            r_acc = (acc - (self.org_acc * 0.9))
            return 1-sum([sum(t) for t in quant_scheme])/96.0 + r_acc
        return (acc - (self.org_acc * 0.9))
        # return (acc - self.org_acc) * 0.1

    def step(self, action):
        action = self._action_wall(action)
        self.quant_scheme.append(action)

        # all the actions are made
        if self.cur_ind == 3:
            # self._final_action_wall()
            assert len(self.quant_scheme) == len(self.layer_feature)

            # acc = 0 # TODO
            acc = lenet_evaluate(self.quant_scheme)
            print("quant_scheme = ", self.quant_scheme)
            print("accuracy = ", acc)

            reward = self.reward(acc)

            w_size = 0# sum([ self.quant_scheme[i]*self.wsize_list[i] for i in range(len(self.quant_scheme)) ])
            w_size_ratio = 0# float(w_size) / float(self.original_wsize)

            info_set = {'w_ratio': w_size_ratio, 'accuracy': acc, 'w_size': w_size}

            if reward > self.best_reward:
                self.best_reward = reward
                prGreen('New best policy: {}, reward: {:.3f}, acc: {:.3f}, w_ratio: {:.3f}'.format(
                    self.quant_scheme, self.best_reward, acc, w_size_ratio))

            obs = self.layer_feature[self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            return obs, reward, done, info_set

        w_size = 0 #sum([ self.quant_scheme[i]*self.wsize_list[i] for i in range(len(self.quant_scheme)) ])
        info_set = {'w_size': w_size}
        reward = 0
        done = False
        self.cur_ind += 1  # the index of next layer
        self.layer_feature[self.cur_ind][-4:] = action
        # build next state (in-place modify)
        obs = self.layer_feature[self.cur_ind, :].copy()
        return obs, reward, done, info_set

    def _action_wall(self, actions):
        assert len(self.quant_scheme) == self.cur_ind

        converted_actions = []
        # limit the action to certain range
        for action in actions:
            action = float(action)
            min_bit, max_bit = self.bound_list[self.cur_ind]
            lbound, rbound = min_bit - 0.5, max_bit + 0.5  # same stride length for each bit
            action = (rbound - lbound) * action + lbound
            action = int(np.round(action, 0))
            converted_actions += [action]
        self.last_action = converted_actions
        return converted_actions  # not constrained here

    def normalize_feature(self, model_info):
        # normalize the state
        model_info = np.array(model_info, 'float')
        print('=> shape of feature (n_layer * n_dim): {}'.format(model_info.shape))
        assert len(model_info.shape) == 2, model_info.shape
        for i in range(model_info.shape[1]):
            fmin = min(model_info[:, i])
            fmax = max(model_info[:, i])
            if fmax - fmin > 0:
                model_info[:, i] = (model_info[:, i] - fmin) / (fmax - fmin)

        return model_info


def train(num_episode, agent, env, output, debug=False):
    # best record
    best_reward = -np.inf
    best_policy = []

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory
    while episode < num_episode:  # counting based on episode
        # reset if it is the start of episode
        print("episode = ", episode)
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if episode <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation, episode=episode)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)

        T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

        # [optional] save intermideate model
        if episode % int(num_episode / 10) == 0:
            agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode
            if debug:
                print('#{}: episode_reward:{:.4f} acc: {:.4f}, weight: {:.4f} MB'.format(episode, episode_reward,
                                                                                         info['accuracy'],
                                                                                         info['w_ratio'] * 1. / 8e6))
            text_writer.write(
                '#{}: episode_reward:{:.4f} acc: {:.4f}, weight: {:.4f} MB\n'.format(episode, episode_reward,
                                                                                     info['accuracy'],
                                                                                     info['w_ratio'] * 1. / 8e6))
            final_reward = T[-1][0]
            # agent observe and update policy
            for i, (r_t, s_t, s_t1, a_t, done) in enumerate(T):
                agent.observe(final_reward, s_t, s_t1, a_t, done)
                if episode > args.warmup:
                    for i in range(args.n_update):
                        agent.update_policy()

            agent.memory.append(
                observation,
                agent.select_action(observation, episode=episode),
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []

            if final_reward > best_reward:
                best_reward = final_reward
                best_policy = env.quant_scheme

            value_loss = agent.get_value_loss()
            policy_loss = agent.get_policy_loss()
            delta = agent.get_delta()
            tfwriter.add_scalar('reward/last', final_reward, episode)
            tfwriter.add_scalar('reward/best', best_reward, episode)
            tfwriter.add_scalar('info/accuracy', info['accuracy'], episode)
            tfwriter.add_scalar('info/w_ratio', info['w_ratio'], episode)
            tfwriter.add_text('info/best_policy', str(best_policy), episode)
            tfwriter.add_text('info/current_policy', str(env.quant_scheme), episode)
            tfwriter.add_scalar('value_loss', value_loss, episode)
            tfwriter.add_scalar('policy_loss', policy_loss, episode)
            tfwriter.add_scalar('delta', delta, episode)
            # record the preserve rate for each layer
            # for i, preserve_rate in enumerate(env.quant_scheme):
            #     tfwriter.add_scalar('preserve_rate_w/{}'.format(i), preserve_rate, episode)

            text_writer.write('best reward: {}\n'.format(best_reward))
            text_writer.write('best policy: {}\n'.format(best_policy))
    text_writer.close()
    return best_policy, best_reward



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Reinforcement Learning')

    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
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
    parser.add_argument('--gpu_id', default='1', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')


    args = parser.parse_args()

    tfwriter = SummaryWriter(logdir=args.output)
    text_writer = open(os.path.join(args.output, 'log.txt'), 'w')

    env = QuantEnv(lenet_info)

    nb_states = env.layer_feature.shape[1]
    nb_actions = 4

    agent = DDPG(nb_states, nb_actions, args)

    best_policy, best_reward = train(args.train_episode, agent, env, args.output, debug=args.debug)
