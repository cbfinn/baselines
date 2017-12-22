import ray
import numpy as np
import tensorflow as tf


@ray.remote
class SingleTaskRunner(object):

    def __init__(self, get_env, make_policy, nsteps, gamma, lam, vf_lr=5e-4, max_grad_norm=None,
            nbatch_train=None, nbatch=None, noptepochs=None):
        self.sess = tf.Session()

        self.env = env = get_env()
        self.policy = make_policy()
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=self.policy.X.dtype.name)
        self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.noptepochs = noptepochs
        self.nbatch = nbatch
        self.nbatch_train = nbatch_train
        self.states = self.policy.initial_state
        self.dones = [False for _ in range(nenv)]

        # form optimizer for training the value function, omitting code with OLDVPRED, since tasks
        # are not consistent across meta-batches.
        R = tf.placeholder(tf.float32, [None])
        A = self.policy.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        vpred = self.policy.vf
        vf_losses1 = tf.square(vpred - R)
        vf_loss = .5 * tf.reduce_mean(vf_losses1)
        vf_params = self.policy.vf_weights
        vf_keys = vf_params.keys()
        vf_params_list = [vf_params[key] for key in vf_keys]
        vf_grads = tf.gradients(vf_loss, vf_params_list)
        if max_grad_norm is not None:
            vf_grads, _vf_grad_norm = tf.clip_by_global_norm(vf_grads, max_grad_norm)
        vf_grads = list(zip(vf_grads, vf_params_list))
        vf_trainer = tf.train.AdamOptimizer(learning_rate=vf_lr, epsilon=1e-5)
        _vf_train = vf_trainer.apply_gradients(vf_grads)

        def vf_train(obs, returns, actions, values):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {self.policy.X:obs, A:actions, ADV:advs, R:returns}
            return self.sess.run([vf_loss, _vf_train], td_map)[:-1]

        self.vf_train = vf_train
        tf.global_variables_initializer().run(session=self.sess) #pylint: disable=E1101

    def reset_task(self):
        # TODO - make sure that if env is a vec env, the model is reset to be the same for all envs
        self.env.reset_model()

    def run(self, pol_weights):
        # obs, returns, actions, values, neglogpacs = runner.run()
        self.policy.assign_sampling_weights(pol_weights, self.sess)

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.policy.step(self.obs, self.sess, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)

        #batch of steps to batch of rollouts
        obs_list = mb_obs
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_values = np.asarray(mb_values, dtype=np.float32)

        # new code: fit value function here
        # compute returns
        mb_returns = np.zeros_like(mb_rewards)
        lastreturn = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
            lastreturn = mb_rewards[t] + self.gamma * nextnonterminal * lastreturn
            mb_returns[t] = lastreturn

        # fit value function to returns
        inds = np.arange(self.nbatch)
        obs, returns, actions, values, neglogpacs = tuple(map(sf01, (mb_obs, mb_returns, mb_actions, mb_values, mb_neglogpacs)))
        for _ in range(self.noptepochs):
            np.random.shuffle(inds)
            for start in range(0, self.nbatch, self.nbatch_train):
                end = start + self.nbatch_train
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in (obs, returns, actions, values))
                self.vf_train(*slices)
        # done fitting value function, record new values
        mb_values = []
        for t in range(self.nsteps):
            _, values, _, _ = self.policy.step(obs_list[t], self.sess, None, mb_dones[t])
            mb_values.append(values)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        # done recording new values.

        last_values = self.policy.value(self.obs, self.sess)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
