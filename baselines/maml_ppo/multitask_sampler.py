import ray
import numpy as np


def multitask_sample(policies, env_specs, nsteps, gamma, lam):
    """
    A function that runs each of the policies in each of the respective environments in parallel.
    Within an environment, the policy is vectorized using N copies of the environment. [Not implemented yet]
    Assumes ray has been initialized.

    Args:
        policies - a list of tensorflow policies
        envs - a list of environment specifications or gym environments
        horizon: how long, per environment, to sample for (# of transitions)

    Returns: a list of dictionaries of sample data, list is over tasks
    """
    return ray.get([singletask_sample.remote(policy, spec, nsteps, gamma, lam) for policy, spec in zip(policies, env_specs)])


@ray.remote
def singletask_sample(model, env_spec, nsteps, gamma, lam):
    """
    Run a policy in the specified environment for the specified horizon of nsteps.
    Note that this does not support LSTM policies

    Args:
        model - a tensorflow policy/value function, a pickleable object with an act(ob, done) function
        env_spec - the specification of the env
        nsteps: how long to collect samples for *per env in the vec env* ( total_num_samples / n_envs )
        gamma & lam: gamma and lambda
    Returns: obs, returns, masks, actions, values, neglogpacs = runner.run()
    """
    # TODO: create vector env using env_spec here (this might be specific to the domain...)

    # starter code
    nenv = env.num_envs
    obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
    obs[:] = env.reset()
    dones = [False for _ in range(nenv)]

    # NOTE: the code below was largely copied from Runner.run() in ppo2/ppo2.py
    mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
    epinfos = []
    for _ in range(nsteps): # TODO - this is the horizon per env in the vector of envs...
        actions, values, neglogpacs = model.step(obs, dones)
        mb_obs.append(obs.copy())
        mb_actions.append(actions)
        mb_values.append(values)
        mb_neglogpacs.append(neglogpacs)
        mb_dones.append(dones)
        obs[:], rewards, dones, infos = env.step(actions)
        for info in infos:
            maybeepinfo = info.get('episode')
            if maybeepinfo: epinfos.append(maybeepinfo)
        mb_rewards.append(rewards)
    # batch of steps to batch of rollouts
    mb_obs = np.asarray(mb_obs, dtype=obs.dtype)
    mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
    mb_actions = np.asarray(mb_actions)
    mb_values = np.asarray(mb_values, dtype=np.float32)
    mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
    mb_dones = np.asarray(mb_dones, dtype=np.bool)
    last_values = model.value(obs, dones)
    # discount/bootstrap off value fn
    mb_returns = np.zeros_like(mb_rewards)
    mb_advs = np.zeros_like(mb_rewards)
    lastgaelam = 0
    for t in reversed(range(nsteps)):
        if t == nsteps - 1:
            nextnonterminal = 1.0 - dones
            nextvalues = last_values
        else:
            nextnonterminal = 1.0 - mb_dones[t+1]
            nextvalues = mb_values[t+1]
        delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_values[t]
        mb_advs[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    mb_returns = mb_advs + mb_values
    return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
        epinfos)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])



