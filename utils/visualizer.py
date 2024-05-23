import numpy as np
import sys
from subprocess import Popen, PIPE # Creates a subprocess, mainly used for GUI application to visualize
import utils # My own module
import visdom


class Visualizer():
    """
    This class includes several functions that can display images and pring logging information
    """

    def __init__(self, configuration):
        """
        Initialize the Visualizer class.

        Input params:
            Configuration -- stores all the configuration
        """
        self.configuration = configuration
        self.display_id = 0
        self.name = configuration['name']

        self.ncols = 0
        self.vis = visdom.Visdom()
        if not self.vis.check_connection():
            self.create_visdom_connections()

    def reset(self):
        pass

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at the default port.
        """
        cmd = sys.executable + ' -m visdom.server'
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """
        Display the current losses on visdom display: dictionary of error labels and values

        Input params:
            epoch: Current epoch
            counter_ratio: Progress (percentage) in the current epoch, bewteen 0 to 1
            losses: Training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'loss_plot_data'):
            self.loss_plot_data = {'X': [], 'Y':[], 'legend': list(losses.keys())}
        self.loss_plot_data['X'].append(epoch + counter_ratio)
        self.loss_plot_data['Y'].append([losses[k] for k in self.loss_plot_data['legend']])
        x = np.squeeze(np.stack([np.array(self.loss_plot_data['X'])] * len(self.loss_plot_data['legend']), 1), axis=1)
        y = np.squeeze(np.array(self.loss_plot_data['Y']), axis=1)
        try:
            self.vis.line(
                X = x,
                Y = y,
                opts = {
                    'title': self.name + ' loss over time',
                    'legend': 'epoch',
                    'xlabel': 'epoch',
                    'ylabel': 'loss'
                },
            win = self.display_id
            )
        except ConnectionError:
            self.create_visdom_connections()

    def plot_current_validation_metrics(self, epoch, metrics):
        if not hasattr(self, 'val_plot_data'):
            self.val_plot_data = {'X': [], 'Y': [], 'legend': list(metrics.keys())}
        self.val_plot_data['X'].append(epoch)
        self.val_plot_data['Y'].append([metrics[k] for k in self.val_plot_data['legend']])
        x = np.squeeze(np.stack([np.array(self.val_plot_data['X'])] * len(self.val_plot_data['legend']), 1), axis=1)
        y = np.squeeze(np.array(self.val_plot_data['Y']), axis=1)
        try:
            self.vis.line(
                X=x,
                Y=y,
                opts={
                    'title': self.name + ' over time',
                    'legend': self.val_plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'metric'},
                win=self.display_id+1)
        except ConnectionError:
            self.create_visdom_connections()        