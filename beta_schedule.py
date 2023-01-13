import numpy as np


class BaseSchedule(object):
    def __init__(self, timesteps=1000, **kwargs):
        self.timesteps = timesteps
        self._init_beta(**kwargs)
        self._init_alpha()
        self._init_alpha_bar()
    
    def __len__(self):
        return self.timesteps
    
    def _init_beta(self):
        raise NotImplementedError
    
    def _init_alpha(self):
        self.alpha = 1 - self.beta
        
    def _init_alpha_bar(self):
        self.alpha_bar = np.cumprod(self.alpha)


class ConstantSchedule(BaseSchedule):
    def _init_beta(self, constant=1e-3):
        self.beta = constant * np.ones(self.timesteps)


class LinearSchedule(BaseSchedule):
    def _init_beta(self, start=1e-4, end=0.01):
        self.beta = np.linspace(start, end, self.timesteps)


class SqrtSchedule(BaseSchedule):
    def _init_beta(self, start=1e-8, end=1e-4):
        self.beta = np.sqrt(np.linspace(start, end, self.timesteps))


class SineSchedule(BaseSchedule):
    def _init_beta(self, scale=0.01):
        start = 1e-4 / scale
        temp = np.linspace(start, 0.5 * np.pi, self.timesteps)
        self.beta = scale * np.sin(temp)


class SigmoidSchedule(BaseSchedule):
    def _init_beta(self, scale=0.01):
        temp = np.linspace(-5, 5, self.timesteps)
        self.beta = scale / (1 + np.exp(-temp))
