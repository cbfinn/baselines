#!/usr/bin/env python
import argparse
import ray
from baselines import bench, logger

def train(env_id, num_timesteps, seed):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.maml_ppo import maml_ppo
    from baselines.maml_ppo.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def get_env():
        def make_env():
            env = gym.make(env_id)
            env = bench.Monitor(env, logger.get_dir())
            return env
        env = DummyVecEnv([make_env])
        env = VecNormalize(env)
        return env

    set_global_seeds(seed)
    policy = MlpPolicy
    ray.init()
    maml_ppo.learn(policy=policy, get_env=get_env, nsteps=2048, nminibatches=64,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1, ent_coef=0.0,
        lr=3e-4, inner_lr=0.1,
        cliprange=0.2,
        ntasks=4,
        total_timesteps=num_timesteps)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='HalfCheetahGoalVel-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()

