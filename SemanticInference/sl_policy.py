from __future__ import division
from __future__ import print_function
import tensorflow as tf
import sys

from networks import policy_nn
from utils import *
from env import Env


relation = sys.argv[1]
# relationSet = ["concept_athletehomestadium", "concept_athleteplaysforteam"]
# relation = "athleteplaysforteam"
graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
# relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'
relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'
print("Relation:", relation)
print("DataPath:", dataPath)
MAX_TRAINING_DATA = 500


class SupervisedPolicy(object):
    """docstring for SupervisedPolicy"""

    def __init__(self, learning_rate=0.001):
        self.initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('supervised_policy'):
            self.state = tf.placeholder(tf.float32, [None, state_dim], name='state')
            self.action = tf.placeholder(tf.int32, [None], name='action')
            self.action_prob = policy_nn(self.state, state_dim, action_space, self.initializer)

            action_mask = tf.cast(tf.one_hot(self.action, depth=action_space), tf.bool)         # Tensor("supervised_policy/Cast:0", shape=(?, 400), dtype=bool)
            self.picked_action_prob = tf.boolean_mask(self.action_prob, action_mask)        # 通过布尔值 过滤元素, mask 为 true 对应的 tensor 的元素

            self.loss = tf.reduce_sum(-tf.log(self.picked_action_prob)) + sum(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='supervised_policy'))       # 统计l2正则化损失
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_prob, {self.state: state})

    def update(self, state, action, sess=None):
        sess = sess or tf.get_default_session()
        _, loss = sess.run([self.train_op, self.loss], {self.state: state, self.action: action})
        return loss


def train():
    tf.reset_default_graph()
    policy_nn = SupervisedPolicy()

    # train_pos     902条三元组用作train_data
    f = open(relationPath)
    train_data = f.readlines()
    f.close()

    num_samples = len(train_data)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if num_samples > MAX_TRAINING_DATA:
            num_samples = MAX_TRAINING_DATA
        else:
            num_episodes = num_samples

        for episode in range(num_samples):
            print("Episode %d" % episode)
            print('Training Sample:', train_data[episode % num_samples][:-1])

            # knowledge graph environment definition
            # Q: 为什么要删除和task:relation相关的三元组
            env = Env(dataPath, train_data[episode % num_samples])
            sample = train_data[episode % num_samples].split()

            # graph.txt中数据为e1,r,e2      kb_env_rl中数据为e1,e2,r
            try:
                good_episodes = teacher(sample[0], sample[1], 5, env, graphpath)
            except Exception as e:
                print('Cannot find a path')
                continue

            for item in good_episodes:
                state_batch = []
                action_batch = []
                for t, transition in enumerate(item):
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)
                state_batch = np.squeeze(state_batch)
                state_batch = np.reshape(state_batch, [-1, state_dim])
                policy_nn.update(state_batch, action_batch)

        saver.save(sess, 'models/policy_supervised_' + relation)
        print('Model saved')


if __name__ == "__main__":
    train()
