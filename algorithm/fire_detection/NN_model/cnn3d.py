import logging
import tensorflow as tf
from easydict import EasyDict
from algorithm.fire_detection.NN_model.libs import Model_skeleton, log_tensor_info
from algorithm.fire_detection.smoke_detection_core.motion_detection import motion_detector_factory


def Cnn3d_conf():
    hparams = EasyDict()

    # Key parameters.
    # Data preprocessing
    hparams.is_standardization = True
    # Classes_num
    hparams.num_classes = 2
    # Epoches for train
    hparams.epoches = 20
    # Batch size
    hparams.batch_size = 64
    # Learning rate
    hparams.learning_rate = 0.001
    # Motion detector.
    hparams.motion_detector = 'background_substraction'

    # The method for adjusting learning rate, which is in ['constant', 'exponential']
    hparams.lr_mode = 'exponential_decay'
    # For exponential decay
    hparams.lr_decay_rate = 0.8
    hparams.lr_decay_steps = 400

    # The following hparams are for optimizer
    # The optimizer for the net, which is in ['sgd', 'mome', 'adam', 'rmsp']
    hparams.optimizer = 'adam'
    # For 'mome'
    hparams.mome_momentum = 0.8
    # For 'adam'
    hparams.adam_beta1 = 0.9
    hparams.adam_beta2= 0.95
    # For 'rmsp'
    hparams.rmsp_mometum = 0.8
    hparams.rmsp_decay = 0.9

    # Weight decay for L1 or L2 or L1_L2 normalization
    # The regularization mode is option in [None, 'L1', 'L2', 'L1_L2']
    hparams.regularization_mode = None
    # For 'L1' or 'L1_L2'
    hparams.L1_scale = 0.0005
    # For 'L2' or 'L1_L2'
    hparams.L2_scale = 0.0005

    # Keep prob for dropout
    hparams.keep_prob = 0.8

    # Epsilon for stability
    hparams.epsilon = 1e-8

    # Gradients will be normalizing to max 10.0. Larger than this value will be clipped.
    hparams.max_grad_norm = 1.0

    return hparams


class Cnn3d(Model_skeleton):
    def __init__(self, hparams):
        super(Cnn3d, self).__init__(hparams=hparams)
        self.motion_detector = motion_detector_factory().get_motion_detector(hparams.motion_detector)
        self.trained_steps = 0
        data_shape = [None, self.hparams.sample_sum_frames, self.hparams.block_size, self.hparams.block_size, 3]
        self.ph_data = tf.placeholder(dtype=tf.float32, shape=data_shape,name='ph_data')
        self.ph_label = tf.placeholder(dtype=tf.float32, shape=[None, self.hparams.num_classes], name='ph_label')
        self.ph_is_training = tf.placeholder(dtype=tf.bool, name='ph_is_training')
        log_tensor_info(self.ph_data)
        log_tensor_info(self.ph_label)
        log_tensor_info(self.ph_is_training)
        logging.info('Model initialization completed!')

        self._add_forward_graph()
        self._add_argmax_output_graph()
        self._add_loss_graph()
        self._add_train_graph()
        self._viz_key_data()
        self._count_trainable_parameters()


    def _add_forward_graph(self):
        l = self._conv3d_layer('conv3d_0', self.ph_data, 16, [3, 3, 3], [1, 1, 1])
        #dense block1
        with tf.variable_scope('block1') as scope:
            for i in range(2):
                l = self._add_layer('dense_layer.{}'.format(i), l, 'A')
            l = self._add_transition('transition1', l, 32)
        # dense block2
        with tf.variable_scope('block2') as scope:
            for i in range(2):
                l = self._add_layer('dense_layer.{}'.format(i), l, 'B')
            l = self._add_transition('transition2', l, 64)
        # dense block3
        with tf.variable_scope('block3') as scope:
            for i in range(2):
                l = self._add_layer('dense_layer.{}'.format(i), l, 'C')
            l = self._conv3d_block('conv_1', l, 128, [1, 1, 1], [1, 1, 1])
            l = tf.reduce_mean(l, axis=[1, 2, 3])
            self.nn_output = self._fc_layer('nn_output', l, hiddens=self.hparams.num_classes)


    def _conv3d_block(self, block_name, input_data, out_channels, kernel, stride, padding='SAME'):
        with tf.variable_scope(block_name):
            conv3d_out = self._conv3d_layer(block_name +'_conv3d', input_data, out_channels=out_channels,
                                            kernel_size=kernel, stride=stride, padding=padding)
            bn_out = self._bn_layer(block_name+'_bn', conv3d_out)
            relu_out = self._relu_layer(block_name+'_relu', bn_out)
            return relu_out

    def _add_layer(self, name, l, type):
        with tf.variable_scope(name) as scope:
            if type == 'A':
                conv3d_S = self._conv3d_block('conv3d_S', l, 12, [1, 3, 3], [1, 1, 1])
                conv3d_T = self._conv3d_block('conv3d_T', conv3d_S, 12, [3, 1, 1], [1, 1, 1])
                c = conv3d_T
            elif type == 'B':
                conv3d_S = self._conv3d_block('conv3d_S', l, 12, [1, 3, 3], [1, 1, 1])
                conv3d_T = self._conv3d_block('conv3d_T', l, 12, [3, 1, 1], [1, 1, 1])
                c = conv3d_S + conv3d_T
            elif type == 'C':
                conv3d_S = self._conv3d_block('conv3d_S', l, 12, [1, 3, 3], [1, 1, 1])
                conv3d_T = self._conv3d_block('conv3d_T', conv3d_S, 12, [3, 1, 1], [1, 1, 1])
                c = conv3d_S + conv3d_T
            elif type == 'D':
                c = self._conv3d_block('conv3d_S', l, 12, [3, 3, 3], [1, 1, 1])
            else:
                c = self._conv3d_block('conv3d_S', l, 12, [3, 3, 3], [1, 1, 1])

            l = tf.concat(axis=4, values=[c, l])
        return l

    def _add_transition(self, name, l, in_channel):
        shape = l.get_shape().as_list()[0:]
        in_channel = shape[4]
        with tf.variable_scope(name) as scope:
            l = self._conv3d_block('conv_1', l, in_channel, [1, 1, 1], [1, 1, 1])
            l = self._maxpool3d_layer('pool', l, [2, 2, 2], [2, 2, 2])

        return l
