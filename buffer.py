import numpy as np
import random

class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 7])

class frame_buffer():
    def __init__(self, buffer_size=1000):
        #self.buffer = []
        self.buffer = np.array([])

        self.buffer_size = buffer_size

    def add(self, frame):
        if self.buffer.shape[0] > 0:
            if self.buffer.shape[2] + 1 > self.buffer_size:
                self.buffer = np.delete(self.buffer, self.buffer_size - 1, axis=2)

        if self.buffer.shape[0] == 0:
            self.buffer = frame
        else:
            self.buffer = np.append(frame, self.buffer, axis=2)

            #if self.buffer.shape[0] + 1 > self.buffer_size:
            #self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
            #self.buffer.append(frame)

    def sample(self, size, skip):
        return [self.buffer[:, :, 0::skip][:,:,0:skip].tolist()]
