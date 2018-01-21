#!/usr/bin/env python3


import copy
import math
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
import os
import pandas as pd
import random
import skimage.color
import skimage.transform
import sys
import tensorflow as tf

from collections import defaultdict
from collections import deque
from ple import PLE
from ple.games.flappybird import FlappyBird
from skimage import img_as_float

from evaluate import evaluate


os.environ["SDL_VIDEODRIVER"] = "dummy"  # this line make pop-out window not appear
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


GAME = 'flappy-bird'    # the name of the game being played for log files
ACTIONS = 2             # number of valid actions
GAMMA = 0.99            # decay rate of past observations
OBSERVE = 10000         # timesteps to observe before training
EXPLORE = 3000000       # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1   # starting value of epsilon
REPLAY_MEMORY = 15000   # number of previous transitions to remember
BATCH = 64              # size of minibatch
FRAME_PER_ACTION = 1

EVALUATE_EVERY_T = 50000
UPDATE_TARGET_EVERY_T = 2000


# for Kaggle score evaluation
class Agent:
    def __init__(self, input_s, action_step):
        self.input_s = input_s
        self.action_step = action_step

    def select_action(self, input_screens, sess):
        feed_screens = self.process_screen(input_screens)
        readout_t = sess.run(self.action_step, feed_dict={self.input_s: [feed_screens]})[0]
        action = np.argmax(readout_t)
        return action

    def process_screen(self, input_screens):
        while len(input_screens) < 4:
            input_screens.append(input_screens[-1])
        feed_screens = np.array(input_screens[:-5:-1])
        feed_screens = feed_screens.transpose([1, 2, 0])
        return feed_screens

    def preprocess(self, screen):
        screen = img_as_float(screen)
        screen = skimage.transform.resize(screen, [80, 80])
        return screen


class Replay_buffer():
    def __init__(self, buffer_size=50000):
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


def make_anim(images, fps=60, true_image=False):
    duration = len(images) / fps

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.fps = fps
    return clip


def createNetwork(name):
    with tf.variable_scope(name):
        # input layer
        s = tf.placeholder(tf.float32, [None, 80, 80, 4])

        # hidden layer
        conv1 = tf.layers.conv2d(
            inputs=s,
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
        dense = tf.layers.dense(
            inputs=flat, units=512, activation=tf.nn.relu)
        Q = tf.layers.dense(
            inputs=dense, units=ACTIONS, activation=None)
        state_dense = tf.layers.dense(
            inputs=flat, units=512, activation=tf.nn.relu)
        state = tf.layers.dense(
            inputs=state_dense, units=1, activation=None)
        readout = state + (Q - tf.reduce_mean(Q, axis=1, keep_dims=True))
        weights = [conv1, pool1, conv2, conv3, flat, dense, Q, state_dense, state, readout]
    return s, readout, Q, weights


def get_update_ops(weights, tweights):
    # return operations assign weight to target network
    src_vars = [v for v in tf.global_variables() if 'online' in v.name]
    tar_vars = [v for v in tf.global_variables() if 'target' in v.name]
    print('Src', src_vars)
    print('Tar', tar_vars)
    update_ops = []
    for src_var, tar_var in zip(src_vars, tar_vars):
        update_ops.append(tar_var.assign(src_var))
    return update_ops
    #  length = len(weights)
    #  update_ops = []
    #  for i in range(length):
    #      if weights[i] in tf.global_variables() and tweights[i] in tf.global_variables():
    #          update_ops.append(tweights[i].assign(weights[i]))
    #  return update_ops


def process_screen(screen, width, height):
    # resize game screen to screen_width x screen_height (default 80 * 80)
    screen = img_as_float(screen)
    screen = skimage.transform.resize(screen, [width, height])
    return screen


def do_action(env, actions):
    for actionid, value in enumerate(actions):
        if int(value) == 1:
            reward = env.act(env.getActionSet()[actionid])
            next_state = env.getScreenGrayscale()
            terminal = env.game_over()
            return next_state, reward, terminal
    print('[!] Something error, action array has no 1.')
    return None, None, None # Shouldn't happen


def trainNetwork(s, readout, h_fc1, ts, treadout, th_fc1, update_ops, sess):
	# define the cost function
    a = tf.placeholder(tf.float32, [None, ACTIONS])
    y = tf.placeholder(tf.float32, [None])
    readout_action = tf.reduce_sum(
        tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cost)
    train_loss = 0.0

    # Redefine origin reward function
    reward_values = {
        "positive": 0.2,  # reward pass a pipe
        "tick": 0.01,     # reward per timestamp
        "loss": -1.0,     # reward of gameover
    }

    # set game environment
    game = FlappyBird()
    env = PLE(game, fps=30, display_screen=False, reward_values=reward_values)
    env.reset_game()

    # record max life for demo video
    max_life_time = 0
    life_time = 0
    #  recorded_frames = []

    # store the previous observations in replay memory
    D = Replay_buffer(REPLAY_MEMORY)

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[1] = 1

    x_t, r_0, terminal = do_action(env, do_nothing)
    x_t = process_screen(x_t, 80, 80)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    #  recorded_frames.append(env.getScreenRGB())

    # saving and loading networks
    saver = tf.train.Saver(max_to_keep=200)
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("checkpoint")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print()
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        print()
    else:
        print()
        print("Could not find old network weights")
        print()

    # start training
    sess.run(update_ops)
    epsilon = INITIAL_EPSILON
    t = 0
    while True:
        if env.game_over():
            # reset game
            env.reset_game()

            # a better result, make demo video
            if life_time > max_life_time:
                max_life_time = life_time
                #  clip = make_anim(recorded_frames, fps=60, true_image=True).rotate(-90)
                #  clip.write_videofile('./clips/demo-video-life-{}.mp4'.format(max_life_time))
                #  clip = None

            print('TIMESTEP', t, '/ Game Over with life time', life_time, '/ max life time', max_life_time)

            # reset counter
            life_time = 0
            #  recorded_frames = []

            # get the first state by doing nothing and preprocess the image to 80x80x4
            do_nothing = np.zeros(ACTIONS)
            do_nothing[1] = 1

            x_t, r_0, terminal = do_action(env, do_nothing)
            x_t = process_screen(x_t, 80, 80)
            s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

            #  recorded_frames.append(env.getScreenRGB())

        # choose an action epsilon greedily
        readout_t = sess.run(readout, feed_dict={s: [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon or t <= OBSERVE:
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[1] = 1  # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_gray, r_t, terminal = do_action(env, a_t)
        x_t1 = process_screen(x_t1_gray, 80, 80)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # add life time and record frame
        life_time += 1
        #  recorded_frames.append(env.getScreenRGB())

        # store the transition in D
        D.add((s_t, a_t, r_t, s_t1, terminal))

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            s_j_batch, a_batch, r_batch, s_j1_batch, t_batch = D.sample(BATCH)

            y_batch = []
            readout_j1_batch = sess.run(treadout, feed_dict={ts: s_j1_batch})
            #  a_max_batch = sess.run(readout, feed_dict={s: s_j1_batch})
            for i in range(0, BATCH):
                terminal = t_batch[i]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
                    #  y_batch.append(r_batch[i] + GAMMA * readout_j1_batch[i][np.argmax(a_max_batch[i])])

            # perform gradient step
            train_loss, _ = sess.run([cost, train_step], feed_dict={
                y: y_batch,
                a: a_batch,
                s: s_j_batch}
            )

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'checkpoint/' + GAME + '-dqn', global_step=t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if t % UPDATE_TARGET_EVERY_T == 0:
            print('TIMESTEP', t, '/ Update Target Agent ({} operations)'.format(len(update_ops)))
            sess.run(update_ops)

        if t % 100 == 0:
            print("TIMESTEP", t, "/ STATE", state,
                "/ EPSILON", epsilon, "/ ACTION", action_index,
                "/ Q_MAX %e" % np.max(readout_t),
                "/ loss %e" % train_loss)

        if t % EVALUATE_EVERY_T == 0 or t == 1:
            agent = Agent(ts, treadout)
            scores = evaluate(agent, sess)
            df = pd.DataFrame({'scores': scores})
            df.to_csv('./scores/score-{}.csv'.format(t), index_label='id')


def tfsession(growth=None, fraction=None):
    if growth is None and fraction is None:
        return tf.Session()
    elif growth is None and fraction is not None:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = fraction
        return tf.Session(config=config)
    elif growth is not None and fraction is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = growth
        return tf.Session(config=config)
    else:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = fraction
        config.gpu_options.allow_growth = growth
        return tf.Session(config=config)


def playGame():
    tf.reset_default_graph()
    sess = tfsession(growth=True)
    print('Build online agent network')
    s, readout, h_fc1, weights = createNetwork('online')
    print('Build target agent network')
    ts, treadout, th_fc1, tweights = createNetwork('target')
    print('Prepare update operations')
    update_ops = get_update_ops(weights, tweights)
    print('Update operations: total {}'.format(len(update_ops)))
    print('Training Network')
    trainNetwork(s, readout, h_fc1, ts, treadout, th_fc1, update_ops, sess)


def main():
    playGame()


if __name__ == "__main__":
    main()
