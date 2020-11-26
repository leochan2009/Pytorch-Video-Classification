# import tensorflow as tf
# import tensorflow.python.ops as ops
import torch
# TODO: pre-compute so that this can be much faster on GPU.
def label_to_levels(label, num_classes):
    #Original code that we are trying to replicate:
    #levels = [1] * label + [0] * (self.num_classes - 1 - label)
    label_vec = torch.ones(1).repeat(torch.squeeze(torch.as_tensor(label)).type(torch.IntTensor))

    # This line requires that label values begin at 0. If they start at a higher
    # value it will yield an error.
    num_zeros = num_classes - 1 - torch.squeeze(torch.as_tensor(label)).type(torch.IntTensor)

    zero_vec = torch.zeros(num_zeros, dtype=torch.float32)

    levels = torch.cat((label_vec, zero_vec), axis=0)

    return levels

# The outer function is a constructor to create a loss function using a certain number of classes.
def OrdinalCrossEntropy(y_true, y_pred, device, num_classes = 4, from_type = 'ordinal_logits'):

    # def __init__(self,
    #              num_classes=4,
    #              importance=None,
    #              from_type="ordinal_logits",
    #              name="ordinal_crossent", **kwargs):
    #     """ Cross-entropy loss designed for ordinal outcomes.
    #
    #     Args:
    #       num_classes: (Optional) how many ranks (aka labels or values) are in the ordinal variable.
    #         If not provided it will be automatically determined by on the prediction data structure.
    #       importance: (Optional) importance weights for each binary classification task.
    #       from_type: one of "ordinal_logits" (default), "logits", or "probs".
    #         Ordinal logits are the output of a CoralOrdinal() layer with no activation.
    #         (Not yet implemented) Logits are the output of a dense layer with no activation.
    #         (Not yet implemented) Probs are the probability outputs of a softmax or ordinal_softmax layer.
    #     """
    #     super(OrdinalCrossEntropy, self).__init__(name=name, **kwargs)
    #
    #     self.num_classes = num_classes
    #
    #     # self.importance_weights = importance
    #     if importance is None:
    #         self.importance_weights = tf.ones(self.num_classes - 1, dtype=tf.float32)
    #     else:
    #         self.importance_weights = tf.cast(importance, dtype=tf.float32)
    #
    #     self.from_type = from_type

        # Ensure that y_true is the same type as y_pred (presumably a float).
        # y_pred = ops.convert_to_tensor_v2(y_pred)
    # y_pred = tf.convert_to_tensor(y_pred)
    # y_true = tf.cast(y_true, y_pred.dtype)

    if num_classes is None:
        # Determine number of classes based on prediction shape.
        if from_type == "ordinal_logits":
            # Number of classes = number of columns + 1
            num_classes = y_pred.shape[1] + 1
        else:
            num_classes = y_pred.shape[1]


    importance_weights = torch.ones(num_classes - 1, dtype=torch.float32)
    # Convert each true label to a vector of ordinal level indicators.
    # TODO: do this outside of the model, so that it's faster?
    tf_levels = []
    for i in range(len(y_true)):
        tf_levels.append(label_to_levels(y_true[i], num_classes))
    tf_levels = torch.stack(tf_levels).to(device)
    if from_type == "ordinal_logits":
        # return ordinal_loss(y_pred, tf_levels, self.importance_weights)
        val = (-torch.sum((torch.nn.functional.logsigmoid(y_pred) * tf_levels
                               # + (tf.math.log_sigmoid(logits) - logits) * (1 - levels)) * tf.convert_to_tensor(importance, dtype = tf.float32),
                               + (torch.nn.functional.logsigmoid(y_pred) - y_pred) * (1 - tf_levels)),
                              axis=1))
        return torch.mean(val)
    elif from_type == "probs":
        raise Exception("not yet implemented")
    elif from_type == "logits":
        raise Exception("not yet implemented")
    else:
        raise Exception("Unknown from_type value " + from_type +
                        " in OrdinalCrossEntropy()")


