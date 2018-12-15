import os

import numpy as np
import click

import keras
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras import metrics
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.engine.topology import Layer
from keras.layers import Activation, Dropout, Flatten, Dense, Input, GlobalAveragePooling2D, Conv2D, BatchNormalization, Reshape, Lambda
from keras.models import Sequential, load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam 
from sklearn.metrics import f1_score
from skmultilearn.model_selection import iterative_train_test_split

from data import DataGenerator, get_data, split_data

np.random.seed(2018)

DEFAULT_IMG_SIZE_WHC = (299, 299, 4)


def f1(y_true, y_pred):
    """
    metric from here
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    """

    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def create_model(num_classes, input_shape = DEFAULT_IMG_SIZE_WHC):
    input_tensor = Input(input_shape)

    base_model = InceptionResNetV2(include_top=False, weights="imagenet", input_tensor=input_tensor, pooling="avg")
    base_model.layers[0].filters += 1
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="sigmoid")()
    # We use linear for focal loss, which already contains a sigmoid.
    # x = Dense(n_out, activation="linear")(x)
        
    return Model(input_tensor, output)


@click.command(help="Create dataset.")
@click.option("-i", "--images-path", prompt=True, type=str)
@click.option("-l", "--labels-path", prompt=True, type=str)
def main(images_path, labels_path):
    keras.backend.clear_session()

    data_df = get_data(images_path, labels_path)

    raw_train, valid =  split_data(data_df)

    model = create_model(num_classes=28, input_shape=input_shape)
    model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["acc", f1])
    # model.compile(loss=[_focal_loss(gamma=2,alpha=0.75)], optimizer=Adam(), metrics=["acc", f1])

    epochs = 50
    batch_size = 64
    checkpointer = ModelCheckpoint("../working/InceptionResNetV2.model", verbose=2, save_best_only=True)
    early_stopping = EarlyStopping(monitor="val_loss", patience=2)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=1, factor=0.1)

    train_generator = DataGenerator.create_train(raw_train, batch_size, DEFAULT_IMG_SIZE_WHC, augument=True)
    validation_generator = DataGenerator.create_train(valid, 100, DEFAULT_IMG_SIZE_WHC, augument=False)

    train_steps = raw_train.shape[0] // batch_size
    valid_steps = valid.shape[0] // batch_size

    # train model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        validation_data=next(validation_generator),
        validation_steps=valid_steps, 
        epochs=epochs, 
        verbose=1,
        callbacks=[checkpointer, reduce_lr],
    )


def _focal_loss(gamma=2, alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


if __name__ == "__main__":
    main()
