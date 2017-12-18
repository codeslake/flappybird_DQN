import gym
import gym_ple
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

import numpy as np
import skimage
import scipy.misc

import traceback
import signal
import os

from utils import *
from buffer import *
from config import config
from model import *
from plot import *

env = gym.make('FlappyBird-v0')

def signal_handler(signal, frame):
    env.close()
    exit()
signal.signal(signal.SIGINT, signal_handler)

def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def train():
    tf.reset_default_graph()
    mainQN = Qnetwork(config.TRAIN.h_size, 'mainQN', False, lr_init=config.TRAIN.lr_init, beta=config.TRAIN.beta1)
    targetQN = Qnetwork(config.TRAIN.h_size, 'targetQN', False)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    trainables = tf.trainable_variables()
    targetOps = updateTargetGraph(trainables, config.TRAIN.tau)

    # Set the rate of random action decrease.
    e = config.TRAIN.startE
    stepDrop = (config.TRAIN.startE - config.TRAIN.endE) / config.TRAIN.annealing_steps

    # create lists to contain total rewards and steps per episode
    rList = []
    avglossList = []

    total_steps = 0

    myBuffer = experience_buffer()
    #frameBuffer = frame_buffer()

    buffer_path = config.TRAIN.path + '/' + tl.global_flag['mode'] + '/buffer'
    ckpt_path = config.TRAIN.path + '/' + tl.global_flag['mode'] + '/checkpoint'

    if config.TRAIN.load_pretrain:
        config.TRAIN.pre_train_steps = 0
        print 'loading memory buffer'
        #myBuffer.buffer = np.load(buffer_path + '/pretrain.npy').tolist()
        myBuffer.buffer = np.load(buffer_path + '/buffer_episode.npy').tolist()
        print len(myBuffer.buffer)

    num_episode = 0 # after pretraining
    total_episode_reward = 0.
    episode_iterations = 0.
    total_iterations = 0.
    report_loss = 0.
    epsilon = 0.
    loss = 0.
    lr = config.TRAIN.lr_init

    try:
        with tf.Session() as sess:
            sess.run(init)

            if config.TRAIN.load_model:
                print('Loading Model...')
                load_model(sess, saver, ckpt_path)

            def run_report():
                print("\n**********REPORT**********")
                print("\tEpisode: %d" % num_episode)
                print("\tEpisode Reward: %.8f" % total_episode_reward)
                print("\tEpisode Steps: %.8f" % episode_iterations)
                print("\tTotal iterations: %d" % total_iterations)
                print("\tBatch Loss: %.8f" % report_loss)
                print("\tEpsilon: %.8f" % epsilon)
                print("\tLearning Rate: %.8f" % lr)

            s, info = env.reset()

            '''
            s = scipy.misc.imresize(s, [84, 84], interp = 'bicubic', mode=None)
            s = np.expand_dims(skimage.color.rgb2grey(s), axis=2)
            for i in range(13):
                frameBuffer.add(s)
            s = frameBuffer.sample(4)

            '''
            s = scipy.misc.imresize(s, [84, 84], interp = 'bicubic', mode=None)
            s = skimage.color.rgb2grey(s)
            s = np.stack((s, s, s, s), axis=2)
            s = np.expand_dims(s, axis=0)

            info = np.stack((info, info, info, info), axis=2)


            if tl.global_flag['plot']:
                graph = plot(config.TRAIN.max_step)
            for i in range(config.TRAIN.num_episodes):
                if total_steps >= config.TRAIN.pre_train_steps:
                    num_episode += 1
                    if num_episode != 0 and (num_episode % config.TRAIN.decay_every == 0):
                        new_lr_decay = config.TRAIN.lr_decay ** (num_episode // config.TRAIN.decay_every)
                        lr = config.TRAIN.lr_init * new_lr_decay
                        if lr < 1e-6:
                            lr = 1e-6
                        sess.run(tf.assign(mainQN.learning_rate, lr))
                    elif num_episode == 0:
                        sess.run(tf.assign(mainQN.learning_rate, config.TRAIN.lr_init))
                        lr = config.TRAIN.lr_init

                episodeBuffer = experience_buffer()
                rAll = 0
                j = 0
                lossList = []
                while j < config.TRAIN.max_step:
                    if tl.global_flag['render'] and num_episode is not 0 and num_episode % 100 == 0:
                        env.render()
                    j += 1
                    if np.random.rand(1) < e or total_steps < config.TRAIN.pre_train_steps:
                        if np.random.randint(0, 20) > 1:
                            a = 1
                        else:
                            a = 0
                        #a = np.random.randint(0, 2)
                    else:
                        a = sess.run(mainQN.predict, feed_dict={mainQN.image: s, mainQN.scalar: info})[0]
                        #a = sess.run(mainQN.predict, feed_dict={mainQN.image: s})[0]


                    s1, r, d, info1 = env.step(a)
                    if r < 0:
                        r = -1
                    elif r > 0:
                        r = 1
                    else:
                        r = 0.1
                    '''
                    if r < 0:
                        r = -1
                    elif r > 0:
                        r = 2
                    else:
                        r = 0
                    '''

                    '''
                    if ai:
                        print '[' + str(total_steps) + '] action:' + str(a) + ' reward: ' + str(r) + '*'
                    else:
                        print '[' + str(total_steps) + '] action:' + str(a) + ' reward: ' + str(r)
                    '''

                    '''
                    s1 = scipy.misc.imresize(s1, [84, 84], interp = 'bicubic', mode=None)
                    s1 = np.expand_dims(skimage.color.rgb2grey(s1), axis=2)
                    frameBuffer.add(s1)
                    s1 = frameBuffer.sample(4)
                    '''
                    s1 = scipy.misc.imresize(s1, [84, 84], interp = 'bicubic', mode=None)
                    s1 = np.expand_dims(np.expand_dims(skimage.color.rgb2grey(s1), axis=2), axis=0)
                    s1 = np.append(s1, s[:, :, :, :3], axis=3)

                    info1 = np.append(np.expand_dims(info1, axis=2), info[:, :, :3], axis=2)

                    total_steps += 1
                    #episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save the experience to our episode buffer.
                    episodeBuffer.add(np.reshape(np.array([s, info, a, r, s1, info1, d]), [1, 7]))  # Save the experience to our episode buffer.
                    #myBuffer.add(np.reshape(np.array([s, info, a, r, s1, info1, d]), [1, 7]))  # Save the experience to our episode buffer.
                    #episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save the experience to our episode buffer.

                    if config.TRAIN.load_pretrain == False and total_steps == config.TRAIN.pre_train_steps:
                        myBuffer.add(episodeBuffer.buffer)
                        np.save(buffer_path + '/pretrain_' + str(i) + '.npy', np.array(myBuffer.buffer))

                    if total_steps > config.TRAIN.pre_train_steps:
                        if e > config.TRAIN.endE:
                            e -= stepDrop
                        else:
                            e = config.TRAIN.endE

                        if total_steps % config.TRAIN.update_freq == 0:
                            trainBatch = myBuffer.sample(config.TRAIN.batch_size)  # Get a random batch of experiences.
                            # Below we perform the Double-DQN update to the target Q-values
                            Q1 = sess.run(mainQN.predict, feed_dict={mainQN.image: np.vstack(trainBatch[:, 4]), mainQN.scalar: np.vstack(trainBatch[:, 5])})
                            Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.image: np.vstack(trainBatch[:, 4]), targetQN.scalar: np.vstack(trainBatch[:, 5])})
                            #Q2 = sess.run(mainQN.Qout, feed_dict={mainQN.image: np.vstack(trainBatch[:, 4])})
                            end_multiplier = -(trainBatch[:, 6] - 1)
                            doubleQ = Q2[range(config.TRAIN.batch_size), Q1]
                            targetQ = trainBatch[:, 3] + (config.TRAIN.y * doubleQ * end_multiplier)

                            # Update the network with our target values.
                            loss, _ = sess.run([mainQN.loss, mainQN.updateModel], feed_dict={mainQN.image: np.vstack(trainBatch[:, 0]), mainQN.scalar: np.vstack(trainBatch[:, 1]), mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:, 2]})
                            lossList.append(loss)

                            updateTarget(targetOps, sess)  # Update the target network toward the primary network.
                    rAll += r
                    s = s1
                    info = info1

                    if d:
                        break

                # reset
                ########
                env.reset()
                total_episode_reward = rAll
                episode_iterations = j
                total_iterations = total_steps
                report_loss = loss
                epsilon = e
                run_report()
                ########

                myBuffer.add(episodeBuffer.buffer)

                if tl.global_flag['plot']:
                    ## plot
                    graph.write_to_handler(graph.step_handler, graph.step_axis, np.append(graph.step_handler.get_xdata(), i), j)

                    # reward
                    rList.append(rAll)
                    if len(rList) > 100:
                        rList = rList[-100:]
                    avg_reward = np.mean(rList)
                    graph.write_to_handler(graph.reward_handler, graph.reward_axis, graph.step_handler.get_xdata(), avg_reward)

                    # loss
                    avg_loss = np.mean(lossList)
                    if total_steps > config.TRAIN.pre_train_steps:
                        avglossList.append(avg_loss)
                    if len(avglossList) > 100:
                        avglossList= avglossList[-100:]
                    avg_loss = np.mean(avglossList)
                    graph.write_to_handler(graph.loss_handler, graph.loss_axis, graph.step_handler.get_xdata(), avg_loss)

                    # e
                    graph.write_to_handler(graph.ep_sub_handler, graph.ep_sub_axis, graph.step_handler.get_xdata(), e)
                    # learning rate
                    graph.write_to_handler(graph.lr_handler, graph.lr_axis, graph.step_handler.get_xdata(), lr)

                    graph.draw()

                # Periodically save the model.
                if num_episode != 0 and num_episode % 1000 == 0:
                    saver.save(sess, ckpt_path + '/model-' + str(num_episode) + '.ckpt')
                    print("Saved Model")
                    np.save(buffer_path + '/buffer_episode', np.array(myBuffer.buffer))
                    print("Saved Buffer")

            saver.save(sess, ckpt_path + '/model-' + str(i) + '.ckpt')
            np.save(buffer_path + '/buffer_episode_'+str(i), np.array(myBuffer.buffer))
    except Exception:
        print(traceback.format_exc())
        env.close()
        exit()

def evaluate():
    with tf.Session() as sess:

        mainQN = Qnetwork(config.TRAIN.h_size, 'mainQN', False)
        saver = tf.train.Saver()

        ckpt_path = config.TRAIN.path + '/' + tl.global_flag['mode'] + '/checkpoint'
        load_model(sess, saver, ckpt_path)

        while True:

            rAll = 0.

            s, info = env.reset()

            s = scipy.misc.imresize(s, [84, 84], interp = 'bicubic', mode=None)
            s = skimage.color.rgb2grey(s)
            s = np.stack((s, s, s, s), axis=2)
            s = np.expand_dims(s, axis=0)

            info = np.stack((info, info, info, info), axis=2)

            while True:
                env.render()
                a = sess.run(mainQN.predict, feed_dict={mainQN.image: s, mainQN.scalar: info})[0]
                s1, r, d, info1 = env.step(a)

                s1 = scipy.misc.imresize(s1, [84, 84], interp = 'bicubic', mode=None)
                s1 = np.expand_dims(np.expand_dims(skimage.color.rgb2grey(s1), axis=2), axis=0)
                s1 = np.append(s1, s[:, :, :, :3], axis=3)

                info1 = np.append(np.expand_dims(info1, axis=2), info[:, :, :3], axis=2)

                rAll += r

                if d:
                    print 'reward: ' + str(r)
                    break;

                s = s1
                info = info1

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='flappy_test', help='model_name')
    parser.add_argument('--is_train', type=str, default='True', help='True or False')
    parser.add_argument('--render', type=str, default='False', help='True or False')
    parser.add_argument('--plot', type=str, default='False', help='True or False')
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['is_train'] = t_or_f(args.is_train)
    tl.global_flag['render'] = t_or_f(args.render)
    tl.global_flag['plot'] = t_or_f(args.plot)

    if not os.path.exists(config.TRAIN.path + '/' + tl.global_flag['mode']):
        os.makedirs(config.TRAIN.path + '/' + tl.global_flag['mode'])
        os.makedirs(config.TRAIN.path + '/' + tl.global_flag['mode'] + '/checkpoint')
        os.makedirs(config.TRAIN.path + '/' + tl.global_flag['mode'] + '/buffer')

    if tl.global_flag['is_train']:
        train()
    else:
        evaluate()
