import os
import time
import joblib
import numpy as np
import os.path as osp
import ray
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance

from baselines.maml_ppo.singletask_runner import SingleTaskRunner

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, ntasks, nbatch_act, nbatch_train, nbatch_pre,
                nsteps, ent_coef, max_grad_norm, inner_lr):
        # nbatch_train - post update batch size
        # nbatch_pre - pre update batch size
        sess = tf.get_default_session()

        make_act_policy = lambda: policy(ob_space, ac_space, nbatch_act, 1, sampling_pol=True)
        train_policy = policy(ob_space, ac_space, nbatch_train, nsteps)

        # TODO - handle different batch sizes for inner and outer
        ob_shape = (ntasks, nbatch_pre,) + ob_space.shape
        TRAIN_X = tf.placeholder(tf.float32, ob_shape)
        ob_shape = (ntasks, nbatch_train,) + ob_space.shape
        VAL_X = tf.placeholder(tf.float32, ob_shape)
        # actions
        TRAIN_A = train_policy.pdtype.sample_placeholder([None, None])
        VAL_A = train_policy.pdtype.sample_placeholder([None, None])
        # advantages
        TRAIN_ADV = tf.placeholder(tf.float32, [None, None])
        VAL_ADV = tf.placeholder(tf.float32, [None, None])

        # old negative log PAC?
        TRAIN_OLDNEGLOGPAC = tf.placeholder(tf.float32, [None, None])
        VAL_OLDNEGLOGPAC = tf.placeholder(tf.float32, [None, None])

        # Adam learning rate
        outer_LR = tf.placeholder(tf.float32, [])  # meta lr, beta
        CLIPRANGE = tf.placeholder(tf.float32, [])

        # pre-update policy
        init_weights = train_policy.pol_weights
        weight_keys = init_weights.keys()
        init_weights_list = [init_weights[key] for key in weight_keys]

        def compute_task_policy(inp):
            """ Compute the updated policy for a single task, analogous to task_metalearn
                in the MAML supervised learning code. Also computes the outer gradient.
            """
            # unpack input:
            T_X, T_A, T_ADV, T_OLDNEGLOGPAC, V_X, V_A, V_ADV, V_OLDNEGLOGPAC = inp

            # get prob distr from pre-update policy
            pre_pi, pre_logstd = train_policy.forward_pol(T_X, init_weights)
            pre_pdparam = tf.concat([pre_pi, pre_pi * 0.0 + pre_logstd], axis=1)
            pre_pd = train_policy.pdtype.pdfromflat(pre_pdparam)

            # code for computing the updated policy weights for a single task starting from init_weights
            train_neglogpac = pre_pd.neglogp(T_A)
            #ratio = tf.exp(T_OLDNEGLOGPAC - train_neglogpac)
            pg_loss = tf.reduce_mean(T_ADV * train_neglogpac)
            # ppo update
            #ratio = tf.exp(T_OLDNEGLOGPAC - train_neglogpac)
            #pg_loss = tf.reduce_mean(-T_ADV * ratio)  # might need to remove reduce_mean
            # remove next two lines to not use clipping
            #pg_losses2 = -T_ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            #pg_loss = tf.reduce_mean(tf.maximum(pg_loss, pg_losses2))
            inner_grad = tf.gradients(pg_loss, init_weights_list)

            gradients = dict(zip(weight_keys, inner_grad))
            # TODO - implement multiple gradient steps
            task_weights = dict(zip(weight_keys, [init_weights[key] - inner_lr*gradients[key] for key in weight_keys]))

            post_pi, post_logstd = train_policy.forward_pol(V_X, task_weights)
            post_pdparam = tf.concat([post_pi, post_pi * 0.0 + post_logstd], axis=1)
            post_pd = train_policy.pdtype.pdfromflat(post_pdparam)

            # compute outer gradient
            val_neglogpac = post_pd.neglogp(V_A)
            ratio = tf.exp(V_OLDNEGLOGPAC - val_neglogpac)
            pg_losses = -V_ADV * ratio
            pg_losses2 = -V_ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            approxkl = .5 * tf.reduce_mean(tf.square(val_neglogpac - V_OLDNEGLOGPAC))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
            entropy = tf.reduce_mean(post_pd.entropy())
            outer_loss = pg_loss - entropy * ent_coef
            # TODO - make this the correct gradient by taking into account theta in the expectation.
            outer_grad = tf.gradients(outer_loss, init_weights_list)
            task_weights_list = [task_weights[key] for key in weight_keys]

            return task_weights_list, outer_grad, outer_loss, entropy, approxkl, clipfrac

        out_dtype = [tf.float32] * len(init_weights_list), [tf.float32] * len(init_weights_list), tf.float32, tf.float32, tf.float32, tf.float32
        inp_packed = (TRAIN_X, TRAIN_A, TRAIN_ADV, TRAIN_OLDNEGLOGPAC,
                VAL_X, VAL_A, VAL_ADV, VAL_OLDNEGLOGPAC)
        result = tf.map_fn(compute_task_policy, elems=inp_packed, dtype=out_dtype, parallel_iterations=ntasks)
        new_task_weights, outer_grads, losses, entropies, approxkls, clipfracs = result

        outer_grads = [tf.reduce_mean(grad, axis=0) for grad in outer_grads]
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(outer_grads, max_grad_norm)
        grads = list(zip(grads, init_weights_list))
        trainer = tf.train.AdamOptimizer(learning_rate=outer_LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        pg_loss = tf.reduce_mean(losses)
        entropy = tf.reduce_mean(entropies)
        approxkl = tf.reduce_mean(approxkls)
        clipfrac = tf.reduce_mean(clipfracs)

        def get_preupdate_weights():
            weights_list = sess.run(init_weights_list)
            return dict(zip(weight_keys, weights_list))

        def get_updated_weights(cliprange, obs, returns, actions, values, neglogpacs):
            """ Return list of weight matrices. """
            advs = returns - values
            # TODO - should do the below on a per-task basis, make sure advs is numtasks x T
            assert advs.shape[0] == ntasks and advs.shape[1] == nbatch_pre
            advs = (advs - advs.mean(axis=1,keepdims=True)) / (advs.std(axis=1,keepdims=True) + 1e-8)
            td_map = {CLIPRANGE:cliprange, TRAIN_X:obs, TRAIN_A:actions, TRAIN_ADV:advs, TRAIN_OLDNEGLOGPAC:neglogpacs}
            new_weights = sess.run(new_task_weights, td_map)
            weight_dicts = []
            for i in range(ntasks):
                wlist = [w[i] for w in new_weights]
                weight_dicts.append(dict(zip(weight_keys, wlist)))
            return weight_dicts

        def train(lr, cliprange, train_obs, train_returns, train_actions, train_values, train_neglogpacs,
                val_obs, val_returns, val_actions, val_values, val_neglogpacs):
            # TODO - should do the below on a per-task basis, make sure advs is numtasks x T
            train_advs = train_returns - train_values
            train_advs = (train_advs - train_advs.mean(axis=-1, keepdims=True)) / (train_advs.std(axis=-1, keepdims=True) + 1e-8)
            val_advs = val_returns - val_values
            val_advs = (val_advs - val_advs.mean(axis=-1, keepdims=True)) / (val_advs.std(axis=-1, keepdims=True) + 1e-8)
            td_map = {TRAIN_X:train_obs, TRAIN_A:train_actions, TRAIN_ADV:train_advs, outer_LR:lr,
                    CLIPRANGE:cliprange, TRAIN_OLDNEGLOGPAC:train_neglogpacs,
                    VAL_X: val_obs, VAL_A: val_actions, VAL_ADV: val_advs,
                    VAL_OLDNEGLOGPAC: val_neglogpacs}
            return sess.run(
                [pg_loss, entropy, approxkl, clipfrac, _train], td_map
                )[:-1]
        self.loss_names = ['policy_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            params = sess.run(init_weights)
            joblib.dump(params, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_policy = train_policy
        self.make_act_policy = make_act_policy
        self.get_updated_weights = get_updated_weights
        self.get_preupdate_weights = get_preupdate_weights
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

def constfn(val):
    def f(_):
        return val
    return f


def learn(*, policy, get_env, ntasks, nsteps, total_timesteps, ent_coef, lr, inner_lr,
            max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0):
    """ Run training.
    Args:
        policy: the policy object
        env: the environment object which encapsulates multiple tasks
        ntasks: the meta-batch size
        nsteps: the number of inner timesteps per task (right now also the number of outer steps per task)
        *If more than one env in the vec env, then this is per env in the vec env!*
        total_timesteps: the total number of steps to run training for
        ent_coef: entropy coefficient
        lr: outer step size
        inner_lr: inner step size

    """
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)
    env = get_env()

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    states = None

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space,
            ntasks=ntasks, inner_lr=inner_lr, nbatch_act=nenvs, nbatch_train=nbatch_train,
            nbatch_pre=nbatch, nsteps=nsteps, ent_coef=ent_coef,
            max_grad_norm=max_grad_norm)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))

    model = make_model()
    # TODO - might be better to have the policy be constructed in the runner process...
    # (i.e. ray might now allow passing the policy object into the constructor)
    runners = [SingleTaskRunner.remote(get_env, model.make_act_policy, nsteps,
        gamma, lam, vf_lr=lr(1), max_grad_norm=max_grad_norm, noptepochs=noptepochs,
        nbatch_train=nbatch_train, nbatch=nbatch) for _ in range(ntasks)]
    #runners = [SingleTaskRunner(env=env, policy=model.make_act_policy, nsteps=nsteps,
    #    gamma=gamma, lam=lam, vf_lr=lr(1), max_grad_norm=max_grad_norm, noptepochs=noptepochs,
    #    nbatch_train=nbatch_train, nbatch=nbatch) for _ in range(ntasks)]

    pre_epinfobuf = deque(maxlen=100)
    post_epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        # sample a new batch of tasks
        for runner in runners:
            runner.reset_task.remote()

        t0 = time.time()
        # gather pre-update data
        preupdate_weights = model.get_preupdate_weights()
        pre_sampling_data = ray.get([runner.run.remote(preupdate_weights) for runner in runners])
        diff = time.time() - t0
        print('Pre sampling time: ' + str(diff))
        #pre_sampling_data = [runner.run(preupdate_weights) for runner in runners]

        # invert list of lists and pack data of new inner list into first dim
        pre_data = [ np.array([task_data[i] for task_data in pre_sampling_data]) for i in range(len(pre_sampling_data[0])) ]
        pre_obs, pre_returns, pre_actions, pre_values, pre_neglogpacs, _, pre_epinfos = pre_data

        # compute updated weights.
        t0 = time.time()
        updated_weights = model.get_updated_weights(cliprange(1.0), pre_obs, pre_returns,
                                                    pre_actions, pre_values, pre_neglogpacs)
        diff = time.time() - t0
        print("Compute theta' time: " + str(diff))

        # gather post-update data
        t0 = time.time()
        post_sampling_data = ray.get([runner.run.remote(weights) for runner, weights in zip(runners, updated_weights)])
        #post_sampling_data = [runner.run(weights) for runner, weights in zip(runners, updated_weights)]
        post_data = [ np.array([task_data[i] for task_data in post_sampling_data]) for i in range(len(post_sampling_data[0])) ]
        post_obs, post_returns, post_actions, post_values, post_neglogpacs, _, post_epinfos = post_data

        diff1 = time.time() - t0
        print("Post sampling time: " + str(diff1))


        pre_epinfobuf.extend(pre_epinfos)
        post_epinfobuf.extend(post_epinfos)
        mblossvals = []
        t0 = time.time()
        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    pre_slices = (pre_obs, pre_returns, pre_actions, pre_values, pre_neglogpacs)
                    post_slices = (arr[:, mbinds] for arr in (post_obs, post_returns, post_actions,
                        post_values, post_neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow,*pre_slices, *post_slices))
        else: # recurrent version
            raise NotImplementedError('Recurrence not currently supported with MAML')
        diff = time.time() - t0
        print("Meta optimization time: " + str(diff))
        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            # TODO - put this back
            #ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch*2)
            logger.logkv("fps", fps)
            #logger.logkv("explained_variance", float(ev))
            logger.logkv('pre_eprewmean', safemean([epinfo['r']  for taskinfo in pre_epinfobuf for epinfo in taskinfo]))
            logger.logkv('pre_eplenmean', safemean([epinfo['l']  for taskinfo in pre_epinfobuf for epinfo in taskinfo]))
            logger.logkv('post_eprewmean', safemean([epinfo['r']  for taskinfo in post_epinfobuf for epinfo in taskinfo]))
            logger.logkv('post_eplenmean', safemean([epinfo['l']  for taskinfo in post_epinfobuf for epinfo in taskinfo]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
