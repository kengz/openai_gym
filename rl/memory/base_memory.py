class Memory(object):

    '''
    The base class of Memory, with the core methods
    '''

    def __init__(self, **kwargs):  # absorb generic param without breaking
        '''Construct externally, and set at Agent.compile()'''
        self.agent = None
        self.state = None

    def reset_state(self, init_state):
        '''reset the state of LinearMemory per episode env.reset()'''
        self.state = init_state

    def add_exp(self, action, reward, next_state, terminal):
        '''add an experience'''
        raise NotImplementedError()

    def get_exp(self, inds):
        '''get a batch of experiences by indices'''
        raise NotImplementedError()

    def pop(self):
        '''get the last experience (batched like get_exp()'''
        raise NotImplementedError()

    def size(self):
        '''get a batch of experiences by indices'''
        raise NotImplementedError()

    def rand_minibatch(self, size):
        '''get a batch of experiences by indices'''
        raise NotImplementedError()
