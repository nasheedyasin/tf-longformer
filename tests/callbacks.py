import six
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging


class HFModelCheckPoint(tf.keras.callbacks.ModelCheckpoint):
    """A custom modelcheckpoint for saving the model int 
    the Huggingface way.
    """

    def __init__(self, * args, **kwargs):
        super(HFModelCheckPoint, self).__init__(* args, **kwargs)

    # Re-implementing the _save_model private method
    def _save_model(self, epoch, logs):
        """Saves the model.
        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            # Block only when saving interval is reached.
            logs = tf_utils.to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)

            try:
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                      logging.warning('Can save best model only with %s available, '
                                      'skipping.', self.monitor)
                    else:
                      if self.monitor_op(current, self.best):
                          if self.verbose > 0:
                              print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                    ' saving model to %s' % (epoch + 1, self.monitor,
                                                             self.best, current, filepath))
                          self.best = current

                          self.model.save_pretrained(filepath)

                      else:
                          if self.verbose > 0:
                              print('\nEpoch %05d: %s did not improve from %0.5f' %
                                    (epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))

                    self.model.save_pretrained(filepath)

                self._maybe_remove_file()
            except IOError as e:
                # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                if 'is a directory' in six.ensure_str(e.args[0]).lower():
                  raise IOError('Please specify a non-directory filepath for '
                                'ModelCheckpoint. Filepath used is an existing '
                                'directory: {}'.format(filepath))
                # Re-throw the error for any other causes.
                raise e

# Test Script
if __name__ == "__main__":
    checkpoint = HFModelCheckPoint('./')