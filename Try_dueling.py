
# coding: utf-8

# In[1]:


# import package needed
import matplotlib.pyplot as plt
import os
os.environ["SDL_VIDEODRIVER"] = "dummy" # make window not appear
import tensorflow as tf
import numpy as np
import math
import skimage.color
import skimage.transform
from ple.games.flappybird import FlappyBird
from ple import PLE
from myUtil import make_anim

import coloredlogs, logging
coloredlogs.install()


# -------------------------------------------- #

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--restore', type=int, default=0, help='episode of checkpoint to restore')
parser.add_argument('--episode', type=int, default=10000, help='episode to train')
parser.add_argument('--store-name', type=str, default='dualing_DQN', help='Name of model to store')

parser.add_argument('--pipe-r', type=float, default=1.0, help='Reward of passing each pipe')
parser.add_argument('--alive-r', type=float, default=0.1, help='Reward of keep alive for t')
parser.add_argument('--die-r', type=float, default=-1.0, help='Reward of dieing')
parser.add_argument('--target-wait', type=int, default=5000, help='How many timestamps to update target agent to online agent')

parser.add_argument('--exploit-show', action='store_true', help='just run and store a video in movie/<store_name>/exploit_show.webm')
parser.add_argument('--evaluate', action='store_true', help='set to evaluate scores')

args = parser.parse_args()

# -------------------------------------------- #

game = FlappyBird()
env = PLE(game, fps=30, display_screen=False) # environment interface to game
env.reset_game()
env.act(0) # dummy input to get screen correct

# get rgb screen
screen = env.getScreenRGB()
# plt.imshow(screen)
print(screen.shape)

# get grayscale screen
# plt.figure()
screen = env.getScreenGrayscale()
# plt.imshow(screen, cmap='gray')
print(screen.shape)
logging.info('test')

# In[2]:


# define input size
screen_width = 80
screen_height = 80
num_stack = 4
def preprocess(screen):
    #screen = skimage.color.rgb2gray(screen)
    screen = skimage.transform.resize(screen, [screen_width, screen_height])
    return screen


import math
import copy
from collections import defaultdict
MIN_EXPLORING_RATE = 10e-4


class Agent:
    def __init__(self, name, num_action, t=0, discount_factor=0.99, store_name='dualing_DQN'):
        self.exploring_rate = 0.1
        self.init_epsilon = 0.1
        self.discount_factor = discount_factor
        self.num_action = num_action
        self.name = name
        with tf.variable_scope(name):
            self.build_model()
            
        self.ckpt_path = os.path.join('./checkpoints/', store_name)
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        self.vars_to_save = [
            var for var in tf.trainable_variables() if name in var.name
        ]
        self.saver = tf.train.Saver(var_list=self.vars_to_save, max_to_keep=100)
    def build_model(self):
        # input: current screen, selected action and reward
        self.input_screen = tf.placeholder(
            tf.float32, shape=[None, screen_width, screen_height, num_stack])
        self.action = tf.placeholder(tf.int32, [None])
        self.reward = tf.placeholder(tf.float32, [None])
        self.is_training = tf.placeholder(tf.bool, shape=[])

        def net(screen, reuse=False):
            with tf.variable_scope(
                    "layers",
                    reuse=reuse,
                    initializer=tf.truncated_normal_initializer(stddev=1e-2)):
                conv1 = tf.layers.conv2d(
                    inputs=screen,
                    filters=32,
                    kernel_size=[8, 8],
                    strides=[4, 4],
                    padding='SAME',
                    activation=tf.nn.relu)
                pool1 = tf.layers.max_pooling2d(
                    conv1, pool_size=[2, 2], strides=[2, 2], padding='SAME')

                conv2 = tf.layers.conv2d(
                    inputs=pool1,
                    filters=64,
                    kernel_size=[4, 4],
                    strides=[2, 2],
                    padding='SAME',
                    activation=tf.nn.relu)
                conv3 = tf.layers.conv2d(
                    inputs=conv2,
                    filters=64,
                    kernel_size=[3, 3],
                    strides=[1, 1],
                    padding='SAME',
                    activation=tf.nn.relu)
                flat = tf.contrib.layers.flatten(conv3)
                print('flat:', flat)
                
                # advantage estimation
                advantage_net = tf.layers.dense(
                    inputs=flat, units=512, activation=tf.nn.relu
                )
                advantage_estimate = tf.layers.dense(
                    inputs=advantage_net,
                    units=self.num_action
                )
                
                # V estimation
                V_net = tf.layers.dense(
                    inputs=flat, units=512, activation=tf.nn.relu
                )
                V_estimate = tf.layers.dense(
                    inputs=V_net,
                    units=1
                )
                
                # combine V(s) and advantage(s, a) to get Q(s, a)
                adv_mean = tf.reduce_mean(
                    advantage_estimate,
                    axis=1,
                    keep_dims=True
                )
                Q = V_estimate + (advantage_estimate - adv_mean)

                return Q

        # optimize
        self.output = net(
            self.input_screen
        )  # Q(s,a,theta) for all a, shape (batch_size, num_action)
        index = tf.stack(
            [tf.range(tf.shape(self.action)[0]), self.action], axis=1)
        self.esti_Q = tf.gather_nd(
            self.output,
            index)  # Q(s,a,theta) for selected action, shape (batch_size, 1)

        self.max_Q = tf.reduce_max(
            self.output, axis=1)  # max(Q(s',a',theta')), shape (batch_size, 1)
        self.tar_Q = tf.placeholder(tf.float32, [None])

        # loss = E[r+max(Q(s',a',theta'))-Q(s,a,theta)]
        self.loss = tf.reduce_mean(
            tf.square(self.reward + self.discount_factor * self.tar_Q -
                      self.esti_Q))

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
        self.g_gvs = optimizer.compute_gradients(
            self.loss,
            var_list=[v for v in tf.global_variables() if self.name in v.name])
        self.train_op = optimizer.apply_gradients(self.g_gvs)
        self.pred = tf.argmax(
            self.output, axis=1
        )  # select action with highest action-value, only used in inference

    def select_action_train(self, input_screen, sess):
        # epsilon-greedy
        if np.random.rand() < self.exploring_rate:
            action = np.random.choice(num_action)  # Select a random action
        else:
            input_screen = np.array(input_screen).transpose([1, 2, 0])
            feed_dict = {
                self.input_screen: input_screen[None, :],
                self.is_training: False,
            }
            action = sess.run(
                self.pred,
                feed_dict=feed_dict)[0]  # Select the action with the highest q
        return action
    
    def select_action(self, input_screen, sess):
        while len(input_screen) < 4:
            input_screen.append(input_screen[-1])
        input_screen = np.array(input_screen[-4:]).transpose([1, 2, 0])
        feed_dict = {
            self.input_screen: input_screen[None, :],
            self.is_training: False,
        }
        action = sess.run(
            self.pred,
            feed_dict=feed_dict)[0]  # Select the action with the highest q
        return action

    def update_policy(self, input_screens, actions, rewards,
                      input_screens_plum, terminal, target_netwrok):
        # use max_Q estimate from target one to update online one
        feed_dict = {
            target_netwrok.input_screen:
            np.array(input_screens_plum).transpose([0, 2, 3, 1]),
            target_netwrok.is_training:
            True,
        }
        max_Q = sess.run(target_netwrok.max_Q, feed_dict=feed_dict)
        max_Q *= ~np.array(terminal)
        feed_dict = {
            self.input_screen: np.array(input_screens).transpose([0, 2, 3, 1]),
            self.tar_Q: max_Q,
            self.action: actions,
            self.reward: rewards,
            self.is_training: True,
        }
        loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

    def update_parameters(self, episode):
        next_epsilon = self.init_epsilon - \
                       (self.init_epsilon - MIN_EXPLORING_RATE) * episode / 100000
        if next_epsilon > MIN_EXPLORING_RATE:
            self.exploring_rate =  next_epsilon

    def shutdown_explore(self):
        # make action selection greedy
        self.exploring_rate = 0
        
    def preprocess(self, screen):
        #screen = skimage.color.rgb2gray(screen)
        screen = skimage.transform.resize(screen, [screen_width, screen_height])
        return screen
    
    def _get_ckpt_name(self, episode):
        return os.path.join(self.ckpt_path, '%s_epidode_%d' % (self.name, episode))
    
    def save_ckpt(self, sess, episode):
        print('[Saver] Saving variables of %s agent(%d episode)... ' % (self.name, episode), end='')
        self.saver.save(sess, self._get_ckpt_name(episode))
        print('Done.')
    
    def load_ckpt(self, sess, episode):
        print('[Saver] Loading variables of %s agent(%d episode)... ' % (self.name, episode), end='')
        self.saver.restore(sess, self._get_ckpt_name(episode))
        print('Done.')


# In[4]:


def get_update_ops():
    # return operations assign weight to target network
    src_vars = [v for v in tf.global_variables() if 'online' in v.name]
    tar_vars = [v for v in tf.global_variables() if 'target' in v.name]
    update_ops = []
    for src_var, tar_var in zip(src_vars, tar_vars):
        update_ops.append(tar_var.assign(src_var))
    return update_ops


def update_target(update_ops, sess):
    sess.run(update_ops)


# In[5]:


# init agent
tf.reset_default_graph()
num_action = len(env.getActionSet())

# agent for frequently updating
online_agent = Agent('online', num_action, store_name=args.store_name)

# agent for slow updating
target_agent = Agent('target', num_action, store_name=args.store_name)
update_ops = get_update_ops()


# In[6]:


class Replay_buffer():
    def __init__(self, buffer_size=45000):
        self.experiences = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.experiences) >= self.buffer_size:
            self.experiences.pop(0)
        self.experiences.append(experience)

    def sample(self, size):
        """
        sameple experience from buffer
        """
        if size > len(self.experiences):
            experiences_idx = np.random.choice(
                len(self.experiences), size=size)
        else:
            experiences_idx = np.random.choice(
                len(self.experiences), size=size, replace=False)
        # from all sampled experiences, extract a tuple of (s,a,r,s')
        screens = []
        actions = []
        rewards = []
        screens_plum = []
        terminal = []
        for i in range(size):
            screens.append(self.experiences[experiences_idx[i]][0])
            actions.append(self.experiences[experiences_idx[i]][1])
            rewards.append(self.experiences[experiences_idx[i]][2])
            screens_plum.append(self.experiences[experiences_idx[i]][3])
            terminal.append(self.experiences[experiences_idx[i]][4])
        return screens, actions, rewards, screens_plum, terminal


# In[ ]:


# init all
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

restore_episode = args.restore
if restore_episode > 0:
    online_agent.load_ckpt(sess, restore_episode)
    update_target(update_ops, sess)

# we can redefine origin reward function
reward_values = {
    "positive": args.pipe_r,  # reward pass a pipe
    "tick": args.alive_r,  # reward per timestamp
    "loss": args.die_r,  # reward of gameover
}
movie_dir = os.path.join('movie', args.store_name)
if not os.path.exists(movie_dir):
    os.makedirs(movie_dir)

def get_reset_environment(reward_values):
    game = FlappyBird()
    env = PLE(
        game,
        fps=30,
        display_screen=False,
        reward_values=reward_values,
        rng=np.random.randint(10000)
    )
    env.reset_game()
    env.act(0)
    return env

if args.exploit_show:
    logging.info('Run a episode without exploration...')
    env = get_reset_environment(reward_values)
    input_screens = [preprocess(env.getScreenGrayscale())] * 4
    display_frames = [env.getScreenRGB()]
    t = 0
    while not env.game_over():
        action = online_agent.select_action(input_screens[-4:], sess)
        reward = env.act(env.getActionSet()[action])
        input_screens.append(preprocess(env.getScreenGrayscale()))
        display_frames.append(env.getScreenRGB())
        t += 1
        if t % 500 == 0: logging.info('Now alive for %d timestep' % t)
        if t >= 12000: break
    logging.info('Alive t = %d' % t)
    clip = make_anim(display_frames, fps=60, true_image=True).rotate(-90)
    clip.write_videofile(os.path.join(movie_dir, "exploit_restore{}_t{}.webm".format(restore_episode, t)), fps=60)
    logging.info('Done')
    exit()

if args.evaluate:
    print('Evaluating...')
    from evaluate import evaluate
    import pandas as pd
    store_dir = os.path.join('./scores/', args.store_name)
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    
    scores = evaluate(online_agent, sess) # evaluate
    scores = pd.DataFrame(scores, columns=['scores'])
    scores.to_csv(os.path.join(store_dir, '%s.csv' % (args.restore)), index_label='id')
    exit()


samples_every_episode = 32
# update_every_episode = 20
update_every_timestep = args.target_wait
print_every_episode = 10
save_video_every_episode = 100
ckpt_every = 40
NUM_EPISODE = args.episode

# init buffer
buffer = Replay_buffer()
game = FlappyBird()
env = PLE(
    game,
    fps=30,
    display_screen=False,
    reward_values=reward_values,
    rng=np.random.RandomState(1))

best_t = 0
cum_t = 0
ts = []

# Adding samples to buffer until it's full enough
ts = []
while len(buffer.experiences) < buffer.buffer_size * 0.8:
    print('The buffer has %d experiences (too few). Adding new ones...' % len(buffer.experiences))
    env.reset_game()
    env.act(0)
    input_screens = [preprocess(env.getScreenGrayscale())] * 4
    t = 0
    while not env.game_over():
        action = online_agent.select_action_train(input_screens[-4:], sess)
        reward = env.act(env.getActionSet()[action])
        input_screens.append(preprocess(env.getScreenGrayscale()))
        buffer.add((input_screens[-5:-1], action, reward, input_screens[-4:],
                    env.game_over()))
        t += 1
    ts.append(t)
logging.info('When adding experiences, average of ts = %.1f, std= %.2f' % (np.mean(ts), np.std(ts)))

# start training
ts = []
for episode in range(restore_episode, restore_episode + NUM_EPISODE + 1):
    # Reset the environment
    env.reset_game()
    env.act(0)  # dummy input to make sure input screen is correct
    
    # record frame
    if episode % save_video_every_episode == 0:
        display_frames = [env.getScreenRGB()]

    # grayscale input screen for this episode
    input_screens = [preprocess(env.getScreenGrayscale())] * 4

    t = 0
    cum_reward = 0
    while not env.game_over():
        # feed four previous screen, select an action
        action = online_agent.select_action_train(input_screens[-4:], sess)
        # execute the action and get reward
        reward = env.act(env.getActionSet()[action])
        # record frame
        if episode % save_video_every_episode == 0:
            display_frames.append(env.getScreenRGB())
        # cumulate reward
        cum_reward += reward
        # append grayscale screen for this episode
        input_screens.append(preprocess(env.getScreenGrayscale()))
        # append experience for this episode
        buffer.add((input_screens[-5:-1], action, reward, input_screens[-4:],
                    env.game_over()))
        
        train_screens, train_actions, train_rewards, \
        train_screens_plum, terminal = buffer.sample(samples_every_episode)
        loss = online_agent.update_policy(
            train_screens, train_actions,
            train_rewards, train_screens_plum,
            terminal, target_agent
        )
        cum_t += 1
        t += 1
        if cum_t >= update_every_timestep:
            cum_t = 0
            update_target(update_ops, sess)

    ts.append(t)
    if t > best_t:
        best_t = t
        print("[Longest So Far] Best t so far: %d at [%d]" % (t, episode))

    # update explore rating and learning rate
    online_agent.update_parameters(episode)
    target_agent.update_parameters(episode)

    if episode % print_every_episode == 0:
        print('To [%d], Recent alive ts: %s,\twhose average=%.1f, std=%.1f' % (episode, ts, np.mean(ts), np.std(ts)))
        ts = []
        print("[{}] time live:{}, cumulated reward: {:.3f},\
        online/target exploring rate: {:.6f}/{:.6f}, loss: {:.6f}".format(
            episode, t, cum_reward, online_agent.exploring_rate, target_agent.exploring_rate, loss))
        
        env.reset_game()
        env.act(0)
        input_screens = [preprocess(env.getScreenGrayscale())] * 4
        t = 0
        while not env.game_over():
            action = online_agent.select_action(input_screens[-4:], sess)
            reward = env.act(env.getActionSet()[action])
            input_screens.append(preprocess(env.getScreenGrayscale()))
            buffer.add((input_screens[-5:-1], action, reward, input_screens[-4:],
                        env.game_over()))
            t += 1
        logging.info('Time live without exploration: %d' % t)

    if episode % save_video_every_episode == 0:  # for every 100 episode, record an animation
        clip = make_anim(display_frames, fps=60, true_image=True).rotate(-90)
        clip.write_videofile(os.path.join(movie_dir, "DQN-{}.webm".format(episode)), fps=60)
    
    if episode % ckpt_every == 0 and episode > restore_episode:
        online_agent.save_ckpt(sess, episode)

print(type(input_screens))

