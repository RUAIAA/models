import tensorflow as tf
from object_detection.utils import ops
from object_detection.utils import shape_utils
from object_detection.utils import static_shape

slim = tf.contrib.slim


MULTI_TASK_LABEL_CLASS_PREDICTIONS = "multi_task_label_class_predictions"

"""Make a separate class for multi-task"""
class MultiTaskLabelConvolutionalClassPredictor(object):
    """MultiTaskLabelConvolutionalBoxPredictor

       Similar to Convolutional BoxPredictor but doesn't handles only classes
       A multi-task label doesn't have a background class and negatives aren't
       backpropagated
    """

    def __init__(self,
                 is_training,
                labels_dict,
                conv_hyperparams,
                use_dropout,
                kernel_size,
                dropout_keep_prob,
                class_prediction_bias_init,
                apply_sigmoid_to_scores):
        """Constructor

        Args:
            is_training: Indicates whether ClassPredictor is in training mode
            labels_dict: a dict mapping label name to number of classes associated with it
        """
        self._is_training = is_training
        self._conv_hyperparams = conv_hyperparams
        self._labels_dict = labels_dict
        self._use_dropout = use_dropout
        self._kernel_size = kernel_size
        self._dropout_keep_prob = dropout_keep_prob
        self._class_prediction_bias_init = class_prediction_bias_init
        self._apply_sigmoid_to_scores = apply_sigmoid_to_scores

    @property
    def labels_dict(self):
        return self._labels_dict

    def predict(self, image_features, num_predictions_per_location, class_predictor_scope):
        net = image_features
        combined_feature_map_shape = shape_utils.combined_static_and_dynamic_shape(
            image_features)
        with tf.variable_scope(class_predictor_scope):
            class_predictors = {}
            with slim.arg_scope(self._conv_hyperparams), \
                 slim.arg_scope([slim.dropout], is_training=self._is_training):
              with slim.arg_scope([slim.conv2d], activation_fn=None,
                                    normalizer_fn=None, normalizer_params=None):
                # Add additional conv layers before the class predictor.
                features_depth = static_shape.get_depth(image_features.get_shape())
                #depth = max(min(features_depth, self._max_depth), self._min_depth)
                for label_name, num_classes in self._labels_dict.items():
                    #if depth > 0 and self._num_layers_before_predictor > 0 :
                    #    net = slim.conv2d(
                    #        net, depth, [1,1], scope=('Conv2d_%d_1x1_%d_')%(i, depth)+label_name)
                    if self._use_dropout:
                        net = slim.dropout(net, keep_prob=self._dropout_keep_prob)
                    class_predictions = slim.conv2d(
                        net, num_predictions_per_location * num_classes,
                        [self._kernel_size, self._kernel_size], scope=class_predictor_scope+"_"+label_name,
                        biases_initializer=tf.constant_initializer(
                            self._class_prediction_bias_init))
                    if self._apply_sigmoid_to_scores:
                        class_predictions = tf.sigmoid(
                            class_predictions_with_background)
                    class_predictions = tf.reshape(
                        class_predictions,
                        tf.stack([combined_feature_map_shape[0],
                                  combined_feature_map_shape[1] *
                                  combined_feature_map_shape[2] *
                                  num_predictions_per_location,
                                  num_classes]))
                    class_predictors.update({label_name: class_predictions})
        return {MULTI_TASK_LABEL_CLASS_PREDICTIONS: class_predictors}
