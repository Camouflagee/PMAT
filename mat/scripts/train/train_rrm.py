import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
# sys.path.append("d:\pythonProject\PMAT\mat")
# sys.path.append("../../")
sys.path.append(str(Path(__file__).resolve().parents[3]))
from mat.config import get_config
from mat.envs.rrm.rrm_env import RRMEnv
from mat.runner.shared.rrm_runner import RRMRunner as Runner
from mat.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = RRMEnv(env_args=all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = RRMEnv(env_args=all_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env

    if all_args.eval_episodes == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.eval_episodes)])

def parse_args(args, parser):
    # 环境相关参数
    parser.add_argument("--n_bs", type=int, default=3)
    parser.add_argument("--n_ue", type=int, default=10) 
    parser.add_argument("--n_rb", type=int, default=4)
    
    # 训练相关参数
    # parser.add_argument("--episode_length", type=int, default=100)
    # parser.add_argument("--n_training_threads", type=int, default=1)
    # parser.add_argument("--n_rollout_threads", type=int, default=32)
    # parser.add_argument("--num_mini_batch", type=int, default=1)
    # parser.add_argument("--num_env_steps", type=int, default=10000000)
    # parser.add_argument("--ppo_epoch", type=int, default=10)
    
    all_args = parser.parse_known_args(args)[0]
    
    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "pmat":
        assert not (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    # cuda设置
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # 初始化环境
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.n_bs

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # 开始训练
    runner = Runner(config)
    runner.run()

    # 环境清理
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

if __name__ == "__main__":
    main(sys.argv[1:])