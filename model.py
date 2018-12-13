import os
import numpy as np
import click

from keras import backend as K
from keras.engine.topology import Layer

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Activation, Dropout, Flatten, Dense, Input, GlobalAveragePooling2D, Conv2D, BatchNormalization, Reshape, Lambda
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import keras
import tensorflow as tf

import pandas as pd

from data import data_generator


def f1(y_true, y_pred):
    '''
    metric from here
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''
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

def f1_loss(y_true, y_pred):
    
    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)


def reduce(x):
    return K.argmax(x, axis=1)

def cast(x):
    return K.cast(x, 'float32')

def create_model(input_shape, n_out):
    inp = Input(input_shape)
    pretrain_model = MobileNetV2(include_top=False, weights=None, input_tensor=inp)
    #x = pretrain_model.get_layer(name="block_13_expand_relu").output
    x = pretrain_model.output
    
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(n_out, activation="relu")(x)
    
    for layer in pretrain_model.layers:
        layer.trainable = True
        
    return Model(inp, x)

@click.command(help="Create dataset.")
@click.option("-i", "--images-path", prompt=True, type=str)
@click.option("-l", "--labels-path", prompt=True, type=str)
def main(images_path, labels_path)
    keras.backend.clear_session()
    data = pd.read_csv(labels_path)

    train_dataset_info = []

    for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
        train_dataset_info.append({
            'path':os.path.join(images_path, name),
            'labels':np.array([int(label) for label in labels])})
    train_dataset_info = np.array(train_dataset_info)


    input_shape = (256,256,3)

    # create train datagen
    train_datagen = data_generator.create_train(
        train_dataset_info, 5, input_shape, augument=True)

    images, labels = next(train_datagen)

    print('min: {0}, max: {1}'.format(images.min(), images.max()))

    model = create_model(input_shape=input_shape, n_out=28)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['acc', f1])

    epochs = 5; batch_size = 64
    checkpointer = ModelCheckpoint('../working/InceptionResNetV2.model', verbose=2, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=1, factor=0.1)

    # split and suffle data 
    np.random.seed(2018)
    indexes = np.arange(train_dataset_info.shape[0])
    np.random.shuffle(indexes)
    train_indexes = indexes[:27500]
    valid_indexes = indexes[27500:]

    train_steps = len(train_indexes)//batch_size
    valid_steps = len(valid_indexes)//batch_size

    # create train and valid datagens
    train_generator = data_generator.create_train(train_dataset_info[train_indexes], batch_size, input_shape, augument=True)
    validation_generator = data_generator.create_train(train_dataset_info[valid_indexes], 100, input_shape, augument=False)

    # train model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        validation_data=next(validation_generator),
        validation_steps=valid_steps, 
        epochs=epochs, 
        verbose=1,
        callbacks=[checkpointer, reduce_lr])


if __name__ == "__main__":
    main()
