import wandb
import numpy as np
import keras

class LogBest(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', records=['val_loss', 'loss', 'val_acc', 'acc']):
        self.monitor = monitor
        self.records = records

        setattr(self, 'best_' + self.monitor, np.inf if 'loss' in self.monitor else 0)
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        if getattr(self, 'best_' + self.monitor) > logs.get(self.monitor): # update
            setattr(self, 'best_' + self.monitor, logs.get(self.monitor))

            log_dict = dict()
            for rs in self.records:
                log_dict['best_' + rs] = logs.get(rs)
            log_dict['best_epoch'] = epoch

            wandb.log(log_dict, commit=False)
