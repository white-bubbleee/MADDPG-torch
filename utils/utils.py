# -*- coding: utf-8 -*-
# @Date       : 2024/5/2218:54
# @Auther     : Wang.zr
# @File name  : utils.py
# @Description:
# -*- coding: utf-8 -*-
# @Date       : 2024/5/2013:32
# @Auther     : Wang.zr
# @File name  : utils.py
# @Description:
import os
import pickle
# import tensorflow as tf
import matplotlib.pyplot as plt
import datetime


def log_reward(reward, episode, file_path):
    # 如果文件存在，则读取现有数据，否则初始化一个空列表
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            try:
                rewards = pickle.load(f)
            except EOFError:
                rewards = []
    else:
        rewards = []
    rewards.append({'episode': episode, 'Reward': reward})
    with open(file_path, 'wb') as f:
        pickle.dump(rewards, f)


def load_pickle(file_path, mode='all'):
    with open(file_path, 'rb') as f:
        rewards = pickle.load(f)
        if mode == 'all':
            return rewards
        elif mode == 'episode':
            return [entry['episode'] for entry in rewards]
        elif mode == 'reward':
            return [entry['Reward'] for entry in rewards]


def init_exp_dir(arglist):
    curtime = datetime.datetime.now()
    curtime_name = f"{curtime.strftime('%Y-%m-%d-%H-%M-%S')}"
    exp_dir = arglist.exp_name + '/' + arglist.scenario + '/' + curtime_name + '/'
    save_exp_dir = arglist.save_dir + exp_dir
    # logs_exp_dir = arglist.save_dir + exp_dir
    results_exp_dir = arglist.plots_dir + exp_dir

    for path in [save_exp_dir, results_exp_dir]:
        os.makedirs(path)

    return {'save_dir': save_exp_dir, 'results_dir': results_exp_dir}


def load_data2_plot(data_path, name: str, show=False):
    # 从.pkl文件中加载列表
    data_list = load_pickle(data_path, mode='reward')

    # 画图
    plt.plot(data_list)
    plt.title(name)
    # plt.show()
    file_dir = os.path.dirname(data_path) + '/'
    plt.savefig(file_dir + name + '.png')
    plt.close()

    if show:
        plt.plot(data_list)
        plt.title(name)
        plt.show()


def save_config_2yaml(arglist, exp_time):
    # 保存配置文件
    save_exp_dir = arglist.save_dir + arglist.exp_name + '/' + exp_time + '/'
    if not os.path.exists(save_exp_dir):
        os.makedirs(save_exp_dir)
    with open(save_exp_dir + 'config.yaml', 'w') as f:
        for key, value in arglist.__dict__.items():
            f.write(f'{key}: {value}\n')


if __name__ == '__main__':
    pass
