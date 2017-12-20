import ray
import numpy as np


def multitask_sample(policies, env_specs, horizon, stochastic):
    """
    A function that runs each of the policies in each of the respective environments in parallel.
    Within an environment, the policy is vectorized using N copies of the environment. [Not implemented yet]
    Assumes ray has been initialized.

    Args:
        policies - a list of tensorflow policies
        envs - a list of environment specifications or gym environments
        horizon: how long, per environment, to sample for (# of transitions)
        stochastic: boolean for whether or not the policy is stochastic

    Returns: a list of dictionaries of sample data, list is over tasks
    """
    return ray.get([singletask_sample.remote(policy, spec, horizon, stochastic) for policy, spec in zip(policies, env_specs)])


@ray.remote
def singletask_sample(policy, env_spec, horizon, stochastic):
    """
    Run a policy in the specified environment for the specified horizon.
    # TODO - vectorize this single task sampler (similar to rllab sampler)

    Args:
        policy - a tensorflow policy, a pickleable object with an act(stochastic, ob) function
        env_spec - the specification of the env
        horizon: how long to collect samples for
        stochastic: whether or not the policy is stochastic
    Returns: dictionary of sample data for this policy and env
    """
    # TODO: create env using env_spec here (this might be specific to the domain...)

    # NOTE: the code below was copied from traj_segment_generator() in ppo1/pposgd_simple.py
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = policy.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

