from easydict import EasyDict as edict

import json

config = edict()
config.TRAIN = edict()
## Adam
config.TRAIN.batch_size = 64
config.TRAIN.lr_init = 1e-5
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 100
# config.TRAIN.lr_decay_init = 0.1
# config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## DQN
config.TRAIN.update_freq = 4
config.TRAIN.y = 0.99
config.TRAIN.startE = 1.
config.TRAIN.endE = 0.0
config.TRAIN.annealing_steps = 200000.
config.TRAIN.num_episodes = 2000000
config.TRAIN.pre_train_steps = 50000
config.TRAIN.h_size = 512
config.TRAIN.tau = 0.01

## TRAIN
config.TRAIN.is_train = False
config.TRAIN.load_pretrain = False
config.TRAIN.load_model = False

config.TRAIN.lr_decay = 0.5
config.TRAIN.decay_every = 3000
config.TRAIN.max_step = 1000
## train set location
#config.TRAIN.path = './saves'
config.TRAIN.path = '/media/junyonglee/Data/saves'

config.VALID = edict()

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
