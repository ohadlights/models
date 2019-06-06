import tensorflow as tf


class EvalHook(tf.train.SessionRunHook):
    def __init__(self):
        super().__init__()
        self._iterations = 0

    def after_run(self, run_context, run_values):
        self._iterations += 1
        if self._iterations % 500:
            tf.logging.info('Eval iteration {}'.format(self._iterations))
