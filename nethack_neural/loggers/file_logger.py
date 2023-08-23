import os
from datetime import datetime
from .abstract_logger import AbstractLogger
import pandas as pd
import matplotlib.pyplot as plt

class FileLogger(AbstractLogger):
    """A logger that logs into two files, one for messages and one for stats.

    Args:
        message_path (str): The path to the file to log messages to.
        stats_path (str): The path to the file to log stats to.
    """
    def __init__(self, message_path, stats_path, name="file_logger", save_plot=False, show_plot=False):
        super().__init__(name)
        self.message_path = message_path
        self.stats_path = stats_path
        self.save_plot = save_plot
        self.show_plot = show_plot
        if os.path.exists(self.message_path):
            os.remove(self.message_path)
        if os.path.exists(self.stats_path):
            os.remove(self.stats_path)

    def log_message(self, time: datetime, msg):
        time = self.format_time(time)
        with open(self.message_path, 'a') as f:
            f.write(f"{time}: {msg}\n")
    
    def log_stats(self, *, time: datetime, timestep, reward, length):
        time = self.format_time(time)
        with open(self.stats_path, 'a') as f:
            f.write(f"{time},{timestep},{reward},{length}\n")

    def save_matplotlib_plot(self):
        """Saves a matplotlib plot of the stats to the stats file path with a .png extension."""
        df = pd.read_csv(self.stats_path, names=['time', 'timestep', 'reward', 'length'])
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        df.plot(x='timestep', y='reward', ax=ax1)
        df.plot(x='timestep', y='length', ax=ax2)
        fig.savefig(self.stats_path + '.png')
        plt.close(fig)

    def show_matplotliib_plot(self):
        """Shows a matplotlib plot of the stats."""
        df = pd.read_csv(self.stats_path, names=['time', 'timestep', 'reward', 'length'])
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        df.plot(x='timestep', y='reward', ax=ax1)
        df.plot(x='timestep', y='length', ax=ax2)
        plt.show()
        plt.close(fig)
            
    def finalize(self):
        if self.save_plot:
            self.save_matplotlib_plot()
        if self.show_plot:
            self.show_matplotliib_plot()