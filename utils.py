import tensorflow as tf
import os

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
        return True
    elif 'FALSE'.startswith(ua):
        return False
    else:
        pass

def load_model(sess, saver, ckpt_dir):

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(ckpt_dir, ckpt_name))
        print '[*] Success to read ' + str(ckpt_name)
    else:
        print '[*] Failed to read checkpoint'
