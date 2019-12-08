import random
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy

import tornado
import logging
import sys
from tornado import options
options.options.mockable()

class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration)
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration)
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration)
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                F.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration)

class TornadoLogger():
    def __init__(self):
        self.logger = logging.getLogger()
        self.formatter = logging.Formatter(
                '[%(levelname)1.1s %(asctime)s.%(msecs)d '
                '%(module)s:%(lineno)d] %(message)s',
                "%Y-%m-%d %H:%M:%S")
        tornado.options.define("access_to_stdout", default=True, help="Log tornado.access to stdout")
        self.bootstrap()

    # tornado server logging
    def init_logging(self, access_to_stdout=False):
        if access_to_stdout:
            access_log = logging.getLogger('tornado.access')
            access_log.propagate = False
            # make sure access log is enabled even if error level is WARNING|ERROR
            access_log.setLevel(logging.INFO)
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(self.formatter)
            access_log.addHandler(stdout_handler)
        
        for handler in self.logger.handlers:  # setting format for all handlers
            handler.setFormatter(self.formatter)

    def bootstrap(self):
        tornado.options.parse_command_line(final=True)
        self.init_logging(tornado.options.options.access_to_stdout)