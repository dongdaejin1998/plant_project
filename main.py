from __future__ import print_function

from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import Hyperband
import IPython


import numpy
from itertools import chain
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers
from keras.applications.resnet_v2 import ResNet50V2

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_curve, auc
matplotlib.use('Agg')

def define_classifier(base_model, nb_unit, nb_FClayer, dr_rate):

    dr_rate = dr_rate * 0.1

    classifier = base_model.output
    classifier = GlobalAveragePooling2D()(classifier)

    if nb_FClayer == 0:
        classifier = Dropout(dr_rate)(classifier)
    elif nb_FClayer == 1:
        classifier = Dense(nb_unit, activation='relu', kernel_initializer='he_uniform')(classifier)
        classifier = Dropout(dr_rate)(classifier)
        classifier = BatchNormalization()(classifier)
    elif nb_FClayer == 2:
        classifier = Dense(nb_unit, activation='relu', kernel_initializer='he_uniform')(classifier)
        classifier = Dropout(dr_rate)(classifier)
        classifier = BatchNormalization()(classifier)
        classifier = Dense(int(nb_unit / 2), activation='relu', kernel_initializer='he_uniform')(classifier)
        classifier = Dropout(dr_rate)(classifier)
        classifier = BatchNormalization()(classifier)
    else:
        classifier = Dense(nb_unit, activation='relu', kernel_initializer='he_uniform')(classifier)
        classifier = Dropout(dr_rate)(classifier)
        classifier = BatchNormalization()(classifier)
        classifier = Dense(nb_unit * 2, activation='relu', kernel_initializer='he_uniform')(classifier)
        classifier = Dropout(dr_rate)(classifier)
        classifier = BatchNormalization()(classifier)
        classifier = Dense(nb_unit, activation='relu', kernel_initializer='he_uniform')(classifier)
        classifier = Dropout(dr_rate)(classifier)
        classifier = BatchNormalization()(classifier)

    out = Dense(1, activation='sigmoid')(classifier)
    model = Model(inputs=base_model.inputs, outputs=out)

    return model

def build_model(hp):
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model = define_classifier(base_model, hp_units, nb_FClayer, dr_rate)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    opti=Adam(lr=hp_learning_rate,decay=decays)
    model.compile(optimizer=opti, loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model

# Constants
nb_unit=3    #nb_unit = 20 * nb_unit
nb_FClayer=3
dr_rate=4    #dr_rate = dr_rate * 0.1
opt='adam'
tf_lrs = 0.01    #learning rate for fitting classifier
tune_lrs = 0.15    #learning rate for fine tunning
decays = 0.01
bs = 32    #batch size
nb_epochs = 20

# Data Path
train_dir = 'D:/disease_plant/gray/train'    #eg: C:/Users/drbon/PycharmProjects/A4/Cancer_data/train'
validation_dir = 'D:/disease_plant/gray/valid'    #eg: 'C:/Users/drbon/PycharmProjects/A4/Cancer_data/validation'
test_dir = 'D:/disease_plant/gray/test'    #eg: 'C:/Users/drbon/PycharmProjects/A4/Cancer_data/test'

# Data Generator
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_it = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), color_mode='rgb', class_mode='binary', batch_size=bs)
val_it = val_datagen.flow_from_directory(validation_dir, target_size=(224, 224), color_mode='rgb', class_mode='binary', batch_size=bs)
test_it = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), color_mode='rgb', class_mode='binary', batch_size=1)

train_filenames = train_it.filenames
nb_train_samples = len(train_filenames)
val_filenames = val_it.filenames
nb_val_samples = len(val_filenames)
test_filenames = test_it.filenames
nb_test_samples = len(test_filenames)

tuner = Hyperband( build_model, # model-building 함수
                       objective = 'val_binary_accuracy', # 최적화 할 objective
                        max_epochs =10, # 테스트할 trials 수
                        factor=3, # 각 trial에 built & fit에 필요한 모델 수
                        directory='my_dir',
                        project_name='intro_to_kt')
class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)

tuner.search(train_it, epochs=10, validation_data=(val_it), callbacks = [ClearTrainingOutput()])
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")
# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
model.fit(train_it, epochs = 10, validation_data = (val_it))


