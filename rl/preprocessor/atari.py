import numpy as np
import scipy as sp
from rl.preprocessor.base_preprocessor import PreProcessor


# Util functions for state preprocessing

def resize_image(im):
    return sp.misc.imresize(im, (110, 84))


def crop_image(im):
    return im[-84:, :]


def process_image_atari(im):
    '''
    Image preprocessing from the paper
    Playing Atari with Deep Reinforcement Learning, 2013
    Takes an RGB image and converts it to grayscale,
    downsizes to 110 x 84
    and crops to square 84 x 84, taking bottomost rows of image
    '''
    im_gray = np.dot(im[..., :3], [0.299, 0.587, 0.114])
    im_resized = resize_image(im_gray)
    im_cropped = crop_image(im_resized)
    return im_cropped


class Atari(PreProcessor):

    '''
    Convert images to greyscale, downsize, crop, then stack 4 states
    NOTE: Image order is cols * rows * channels to match openai gym format
    Input to model is rows * cols * channels (== states)
    '''

    def __init__(self, **kwargs):  # absorb generic param without breaking):
        super(Atari, self).__init__()

    def preprocess_state(self):
        processed_state_queue = (
            process_image_atari(self.state),
            process_image_atari(self.previous_state),
            process_image_atari(self.pre_previous_state),
            process_image_atari(self.pre_pre_previous_state))
        processed_state = np.stack(processed_state_queue, axis=-1)
        return processed_state

    def preprocess_memory(self, action, reward, next_state, done):
        self.add_raw_exp(action, reward, next_state, done)
        if (self.exp_queue_size() < self.MAX_QUEUE_SIZE):  # insufficient queue
            return
        (_state, action, reward, next_state, done) = self.exp_queue[-1]
        processed_next_state_queue = (
            process_image_atari(self.exp_queue[-1][3]),
            process_image_atari(self.exp_queue[-2][3]),
            process_image_atari(self.exp_queue[-3][3]),
            process_image_atari(self.exp_queue[-4][3]))
        processed_state = self.preprocess_state()
        processed_next_state = np.stack(processed_next_state_queue, axis=-1)
        self.debug_state(processed_state, processed_next_state)
        processed_exp = (action, reward, processed_next_state, done)
        return processed_exp
