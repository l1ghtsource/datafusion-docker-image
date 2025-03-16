from transformers import TrainerCallback
import time


class TimeLimitCallback(TrainerCallback):
    def __init__(self, time_limit_hours):
        self.time_limit_seconds = time_limit_hours * 3600
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.time_limit_seconds:
            control.should_training_stop = True