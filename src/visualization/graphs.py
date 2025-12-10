'''# src/visualization/graphs.py
import collections
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class LiveGraph:
    def __init__(self, max_len=150):
        self.max_len = max_len
        self.bpm_data = collections.deque(maxlen=max_len)
        self.stress_data = collections.deque(maxlen=max_len)

        plt.style.use("ggplot")
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.ax1.set_title("Heart Rate (BPM)")
        self.ax2.set_title("Stress Index")

    def update(self, i):
        self.ax1.cla()
        self.ax2.cla()
        self.ax1.plot(list(self.bpm_data))
        self.ax2.plot(list(self.stress_data))
        self.ax1.set_ylim(40, 150)
        self.ax2.set_ylim(0, 100)

    def add_values(self, bpm, stress):
        if bpm is not None:
            self.bpm_data.append(bpm)
        if stress is not None:
            self.stress_data.append(stress)

    def start(self):
        ani = animation.FuncAnimation(self.fig, self.update, interval=200)
        plt.show()
'''

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import collections

_anim_ref = None     # prevent GC

class LiveGraph:
    def __init__(self):
        self.bpm_history = collections.deque(maxlen=200)
        self.stress_history = collections.deque(maxlen=200)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.ax1.set_title("Heart Rate (BPM)")
        self.ax2.set_title("Stress Index")

    def add_values(self, bpm, stress):
        if bpm is not None:
            self.bpm_history.append(bpm)
        if stress is not None:
            self.stress_history.append(stress)

    def update(self, frame):
        self.ax1.cla()
        self.ax2.cla()

        self.ax1.set_title("Heart Rate (BPM)")
        self.ax2.set_title("Stress Index")

        if len(self.bpm_history) > 0:
            self.ax1.plot(list(self.bpm_history))

        if len(self.stress_history) > 0:
            self.ax2.plot(list(self.stress_history))

    def start(self):
        global _anim_ref

        _anim_ref = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=300
        )

        plt.show(block=True)
