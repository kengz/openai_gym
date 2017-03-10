import numpy as np
from rl.preprocessor.base_preprocessor import PreProcessor


class NoPreProcessor(PreProcessor):

    '''
    Default class, no preprocessing
    '''

    def __init__(self, **kwargs):  # absorb generic param without breaking):
        super(NoPreProcessor, self).__init__()

    def preprocess_state(self):
        return self.state

    def preprocess_memory(self, action, reward, next_state, done):
        '''No state processing'''
        self.add_raw_exp(action, reward, next_state, done)
        (_state, action, reward, next_state, done) = self.exp_queue[-1]
        processed_exp = (action, reward, next_state, done)
        return processed_exp


class StackStates(PreProcessor):

    '''
    Current and last state are concatenated to form input to model
    '''

    def __init__(self, **kwargs):  # absorb generic param without breaking):
        super(StackStates, self).__init__(max_queue_size=2)

    def preprocess_state(self):
        processed_state = np.concatenate([self.previous_state, self.state])
        return processed_state

    def preprocess_memory(self, action, reward, next_state, done):
        '''Concatenate: previous + current states'''
        self.add_raw_exp(action, reward, next_state, done)
        if (self.exp_queue_size() < self.MAX_QUEUE_SIZE):  # insufficient queue
            return
        (state, action, reward, next_state, done) = self.exp_queue[-1]
        processed_state = self.preprocess_state()
        processed_next_state = np.concatenate([state, next_state])
        self.debug_state(processed_state, processed_next_state)
        processed_exp = (action, reward, processed_next_state, done)
        return processed_exp


class DiffStates(PreProcessor):

    '''
    Different between current and last state is input to model
    '''

    def __init__(self, **kwargs):  # absorb generic param without breaking):
        super(DiffStates, self).__init__(max_queue_size=2)

    def preprocess_state(self):
        processed_state = self.state - self.previous_state
        return processed_state

    def preprocess_memory(self, action, reward, next_state, done):
        '''Change in state, curr_state - last_state'''
        self.add_raw_exp(action, reward, next_state, done)
        if (self.exp_queue_size() < self.MAX_QUEUE_SIZE):  # insufficient queue
            return
        (state, action, reward, next_state, done) = self.exp_queue[-1]
        processed_state = self.preprocess_state()
        processed_next_state = next_state - state
        self.debug_state(processed_state, processed_next_state)
        processed_exp = (action, reward, processed_next_state, done)
        return processed_exp
