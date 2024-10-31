# MADDPG-torch仓库
基于pytroch实现的多智能体强化学习算法MADDPG

如果需要tensorflow版本的代码，见我的[MADDPG-tf2仓库](https://github.com/white-bubbleee/MADDPG-tf2)
## 0.环境配置
### conda环境配置
 - `python==3.9`
 - `torch==1.13.1+cu116` 根据自己电脑的CUDA版本选择
 - `gym==0.20.0`

### 多智能体强化学习仿真环境配置
在上述建立的conda环境中，到[仿真环境github地址](https://github.com/openai/multiagent-particle-envs)下载代码到工程文件夹中，`cd multiagent-particle-envs`使用`pip install -e .`安装multiagent-particle-envs


## 1.MADDPG算法
运行`train_maddpg.py`

如果需要修改参数，修改`base\args_config.py`文件的`parse_args_maddpg`函数的里面定义的参数。


