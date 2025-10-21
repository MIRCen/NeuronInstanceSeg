from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
import os

bn_axis = 3
channel_axis = bn_axis

pretrained_model = '../models_pretrained'
densenet121_weight = os.path.join(pretrained_model, 'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
inception_resnet_v2_weight = os.path.join(pretrained_model, 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5')

def schedule_steps(epoch, steps):
    for step in steps:
        if step[1] > epoch:
            print("Setting learning rate to {}".format(step[0]))
            return step[0]
    print("Setting learning rate to {}".format(steps[-1][0]))
    return steps[-1][0]
        
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return 1 - (dice_coef(y_true, y_pred))

def softmax_dice_loss(y_true, y_pred):
    # return categorical_crossentropy(y_true, y_pred) * 0.6 + dice_coef_loss(y_true[..., 0], y_pred[..., 0]) * 0.2 + dice_coef_loss(y_true[..., 1], y_pred[..., 1]) * 0.2
    return categorical_crossentropy(y_true, y_pred) * 0.5 + dice_coef_loss(y_true[..., 0], y_pred[..., 0]) * 0.2 + dice_coef_loss(y_true[..., 1], y_pred[..., 1]) * 0.3

def dice_coef_rounded_ch0(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_rounded_ch1(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 1]))
    y_pred_f = K.flatten(K.round(y_pred[..., 1]))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)
