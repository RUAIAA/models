"""Function to build class predictor from configuration."""

from object_detection.core import class_predictor
from object_detection.protos import class_predictor_pb2


def build(argscope_fn, class_predictor_config, is_training, labels_dict):
  """Builds box predictor based on the configuration.

  Builds box predictor based on the configuration. See box_predictor.proto for
  configurable options. Also, see box_predictor.py for more details.

  Args:
    argscope_fn: A function that takes the following inputs:
        * hyperparams_pb2.Hyperparams proto
        * a boolean indicating if the model is in training mode.
      and returns a tf slim argscope for Conv and FC hyperparameters.
    box_predictor_config: box_predictor_pb2.BoxPredictor proto containing
      configuration.
    is_training: Whether the models is in training mode.
    num_classes: Number of classes to predict.

  Returns:
    box_predictor: box_predictor.BoxPredictor object.

  Raises:
    ValueError: On unknown box predictor.
  """
  if not isinstance(class_predictor_config, class_predictor_pb2.ClassPredictor):
    raise ValueError('class_predictor_config not of type '
                     'class_predictor_pb2.ClassPredictor.')

  class_predictor_oneof = class_predictor_config.WhichOneof('class_predictor_oneof')

  if  class_predictor_oneof == 'multi_task_label_convolutional_class_predictor':
    multi_task_label_class_predictor = class_predictor_config.multi_task_label_convolutional_class_predictor
    conv_hyperparams = argscope_fn(multi_task_label_class_predictor.conv_hyperparams,
                                   is_training)
    class_predictor_object = class_predictor.MultiTaskLabelConvolutionalClassPredictor(
        is_training=is_training,
        labels_dict=labels_dict,
        conv_hyperparams=conv_hyperparams,
        use_dropout=multi_task_label_class_predictor.use_dropout,
        dropout_keep_prob=multi_task_label_class_predictor.dropout_keep_probability,
        kernel_size=multi_task_label_class_predictor.kernel_size,
        apply_sigmoid_to_scores=multi_task_label_class_predictor.apply_sigmoid_to_scores,
        class_prediction_bias_init=multi_task_label_class_predictor.class_prediction_bias_init
    )
    return class_predictor_object
