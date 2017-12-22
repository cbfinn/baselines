import numpy as np
import tensorflow as tf
from baselines.a2c.utils import ortho_init
from baselines.common.distributions import make_pdtype


class MlpPolicy(object):

    def __init__(self, ob_space, ac_space, nbatch, nsteps, sampling_pol=False): #pylint: disable=W0613
        """
        Args:
            weights: weights to be used, if another policy has already been constructed.
            sampling_pol: If True, then this policy is for sampling only.
        """
        ob_shape = (nbatch,) + ob_space.shape
        self.actdim = ac_space.shape[0]

        self.pol_weights, self.vf_weights = self.construct_weights()

        if sampling_pol:
            self.X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs

            # placeholders and assign ops for assigning weights
            self.weight_placeholders = {key: tf.placeholder(self.pol_weights[key].get_shape()) for key in self.weights}
            self.assign_weight_ops = [tf.assign(self.pol_weights[key], self.weight_placeholders[key]) for key in self.pol_weights]
            self.pi, logstd = self.forward_pol(X, self.pol_weights)
            self.vf = self.forward_vf(X, self.vf_weights)
            self.pdtype = make_pdtype(ac_space)
            pdparam = tf.concat([self.pi, self.pi * 0.0 + logstd], axis=1)
            self.pd = self.pdtype.pdfromflat(pdparam)
            a0 = self.pd.sample()
            neglogp0 = self.pd.neglogp(a0)

            def step(ob, sess, *_args, **_kwargs):
                a, v, neglogp = sess.run([a0, self.vf, neglogp0], {X:ob})
                return a, v, self.initial_state, neglogp

            def value(ob, sess, *_args, **_kwargs):
                return sess.run(self.vf, {X:ob})
            self.step = step
            self.value = value

        else:
            self.X = None
            self.weight_placeholders = None

        self.initial_state = None

    def assign_sampling_weights(self, new_weights, sess):
        if self.weight_placeholders == None:
            print 'ERROR - this policy is not a sampling policy'
        feed_dict = {self.weight_placeholders[key]: new_weights[key] for key in new_weights}
        sess.run(self.assign_weight_ops, feed_dict)

    def construct_policy_weights(self, scope):
        """ construct symbolic weights for the policy"""
        init_scale = np.sqrt(2)
        get_var = tf.get_variable

        with tf.variable_scope(scope, reuse=None):
            w1 = get_var("w1", [self.X.get_shape()[1], 64], initializer=ortho_init(init_scale))
            b1 = get_var("b1", [64], initializer=tf.constant_initializer(0.0))
            w2 = get_var("w2", [64, 64], initializer=ortho_init(init_scale))
            b2 = get_var("b2", [64], initializer=tf.constant_initializer(0.0))
            w3 = get_var("w3", [64, self.actdim], initializer=ortho_init(0.01))
            b3 = get_var("b3", [self.actdim], initializer=tf.constant_initializer(0.0))

            logstd = tf.get_variable(name="logstd", shape=[1, self.actdim],
                initializer=tf.zeros_initializer())
        weights = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2, 'w3': w3, 'b3': b3, 'logstd': logstd}


    def construct_weights(self, scope):
        """ construct symbolic weights for the value function and policy"""
        init_scale = np.sqrt(2)
        get_var = tf.get_variable

        weights = self.construct_policy_weights()
        with tf.variable_scope(scope, reuse=None):
            vw1 = get_var("vw1", [self.X.get_shape()[1], 64], initializer=ortho_init(init_scale))
            vb1 = get_var("vb1", [64], initializer=tf.constant_initializer(0.0))
            vw2 = get_var("vw2", [64, 64], initializer=ortho_init(init_scale))
            vb2 = get_var("vb2", [64], initializer=tf.constant_initializer(0.0))
            vw3 = get_var("vw3", [64, 1], initializer=ortho_init(1.0))
            vb3 = get_var("vb3", [1], initializer=tf.constant_initializer(0.0))

        vf_weights = {'vw1': vw1, 'vb1': vb1, 'vw2': vw2, 'vb2': vb2, 'vw3': vw3, 'vb3': vb3}
        return weights, vf_weights

    def forward_pol(self, inp, weights):
        """ symbolic forward pass through both policy and value function """
        act = tf.tanh

        z = tf.matmul(inp, weights['w1']) + weights['b1']
        h = act(z)
        z = tf.matmul(h, weights['w2']) + weights['b2']
        h = act(z)
        action = tf.matmul(h, weights['w3']) + weights['b3']  # was self.pi

        logstd = weights['logstd']

        return action, logstd

    def forward_vf(self, inp, weights):
        """ symbolic forward pass through the value function """
        act = tf.tanh

        z = tf.matmul(inp, weights['vw1']) + weights['vb1']
        h = act(z)
        z = tf.matmul(h, weights['vw2']) + weights['vb2']
        h = act(z)
        value = tf.matmul(h, weights['vw3']) + weights['vb3']  # was self.vf

        return value[:, 0]

