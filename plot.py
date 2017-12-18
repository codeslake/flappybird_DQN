import matplotlib.pyplot as plt
import numpy as np
plt.ion()

class plot():
    def __init__(self, max_steps):
        plt.show(block=False)
        self.fig = plt.gcf()
        self.fig.set_size_inches(10, 12)
        self.fig.show()

        # Steps
        self.step_axis, self.step_handler = self.create_graph(self.fig, 411, None, 'steps', 'k-')

        # rewards
        self.reward_axis, self.reward_handler = self.create_graph(self.fig, 412, None, 'reward', 'r-')

        # loss
        self.loss_axis, self.loss_handler = self.create_graph(self.fig, 413, None, 'loss', 'b-')
        self.ep_sub_axis, self.ep_sub_handler = self.create_sub_graph(self.loss_axis, 'epsilon', 'g-', y_lim=[0, 1])

        # lr
        self.lr_axis, self.lr_handler = self.create_graph(self.fig, 414, 'episodes', 'learning rate', 'b-')

    def draw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def create_graph(self, fig, subplot_point, x_label, y_label, color, autoscale=True, y_lim = None):
        axis = fig.add_subplot(subplot_point)
        axis.set_autoscaley_on(autoscale)
        axis.grid()
        #axis.set_xlim()
        if y_lim is not None:
            axis.set_ylim(y_lim)

        if x_label is not None:
            axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        handler, = axis.plot([], [], color)

        return axis, handler

    def create_sub_graph(self, axis, y_label, color, y_lim = None):
        sub_axis = axis.twinx()
        sub_axis.set_ylabel(y_label)

        if y_lim is not None:
            sub_axis.set_ylim(y_lim)

        handler, = sub_axis.plot([], [], color)

        return sub_axis, handler

    def write_to_handler(self, handler, axis, x_data, y_data):
        handler.set_xdata(x_data)
        handler.set_ydata(np.append(handler.get_ydata(), y_data))
        axis.relim()
        axis.autoscale_view()
