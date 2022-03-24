import scipy.io as sio
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scikitplot as skplt
import time
import os
import math
from keras.applications.resnet_v2 import preprocess_input
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import random
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

# command to install extra packages: pip install pandas==1.2.2 scikit-plot==0.3.7

data_path = '.\\FlowerData'
test_images_indices = list(range(301,473))

# supress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# set random seeds
np.random.seed(10)
random.seed(10)

params = {
    "model": {
        "learning_rate": 5e-3,
        "batch_size": 32,
        "epochs": 5,
        "dense_layer_width": 64,
        "regularization": 0.01,
        "freeze_layers_until": 178,
        "dropout_rate": 0.5
    },
    "split_data": {
        "valid_percent": 0.20,
    }
}

def main():
    start_time = time.perf_counter()
    params["split_data"]["valid_percent"] = 0 # use all data for final training
    images, labels = splitData(*loadData(), params["split_data"])
    model = createModel(**params["model"])
    train_images_amount, train_images_flower_amount = train(
        model,
        params["model"],
        images["train"],
        labels["train"],
    )
    probas = test(images["test"], model)
    results = evaluate(probas, labels["test"], start_time)
    report(*results, model, train_images_amount, train_images_flower_amount)

## used for grid searching parameters
def grid_search_tune():
    params["split_data"]["valid_percent"] = 0 # grid search implements k-fold so no need to split data here
    images, labels = splitData(*loadData(), params["split_data"])
    model = create_scikit_model()
    grid_search(model, images["train"], labels["train"])

## used for evaluating a model one time during tuning
def three_fold():
    params["split_data"]["valid_percent"] = 0 # using k-fold so no need for validation set
    images, labels = splitData(*loadData(), params["split_data"])
    model = create_scikit_model()
    kfold(model, images["train"], labels["train"])

def loadData():
    """Loads and preprocesses data for resnet_v2 network

    Returns:
        images (numpy array of images of dimensions (224, 224, 3)): training images
        labels (numpy array of integers 0/1): training labels
    """    

    print("loading data")
    # load images and labels
    imagesAndLabels = sio.loadmat(os.path.join(data_path, 'FlowerDataLabels'))
    images = imagesAndLabels['Data'][0]
    labels = imagesAndLabels['Labels'][0]

    # preprocess images for resnet_v2 network
    resized_images = [preprocess_input(cv2.resize(im, (224, 224))) for im in images]

    return np.array(resized_images).astype("float32"), np.array(labels)

def splitData(images, labels, params):
    """Splits images and labels into three sets, train, validation & test

    Args:
        images: numpy array of images
        labels: numpy array of labels
        params: parameters for splitting data
            valid_percent: percentage of training set in validation set

    Returns:
        images: dict with three keys, "train", "validation" & "test" containing the images for each set.
        labels: dict with three keys, "train", "validation" & "test" containing the labels for each set.
    """    
    # image indexes specified in test_images_indices start from 1
    all_indexes = list(range(1, len(labels) + 1))

    # all indexes not specified for test, are used for train. (subtract 1 to get array index)
    train_indexes = np.array([(idx - 1) for idx in all_indexes if idx not in test_images_indices])
    np.random.shuffle(train_indexes)

    train_image_amount = int(len(train_indexes)*(1-params["valid_percent"]))

    train_images = [images[i] for i in train_indexes[:train_image_amount]]
    train_labels = [labels[i] for i in train_indexes[:train_image_amount]]

    validation_images = [images[i] for i in train_indexes[train_image_amount:]]
    validation_labels = [labels[i] for i in train_indexes[train_image_amount:]]
    
    test_images = [images[i - 1] for i in test_images_indices]
    test_labels = [labels[i - 1] for i in test_images_indices]

    images = {
        "train": np.array(train_images),
        "validation": np.array(validation_images),
        "test": np.array(test_images)
    }

    labels = {
        "train": np.array(train_labels),
        "validation": np.array(validation_labels),
        "test": np.array(test_labels)
    }
    return images, labels

def createModel(learning_rate = 5e-3, dense_layer_width = 64, regularization = 0.01, dropout_rate = 0.5, freeze_layers_until = 178, **kwargs):
    """Creates a new binary classification model built upon ResNet50V2.
    The top is removed and extra layers are added to adapt it for binary classification.

    Args:
        params: dict of model parameters: "learning_rate", "dense_layer_width", "regularization", freeze_layers_until
    Returns:
        model: Keras Model class.
    """
    # we need to import tensflow out of the global scope to avoid a pickling error when running parallel jobs
    import tensorflow as tf
    tf.random.set_seed(10)

    print("initializing model")

    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dense(dense_layer_width, activation='relu', kernel_regularizer=l2(regularization)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(regularization))
    ])  

    # freeze most layers in base model
    for layer in base_model.layers[:freeze_layers_until]:
        layer.trainable = False
        
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy'],
    )

    return model

def train(model, params, images, labels, validation_images = [], validation_labels = []):
    """trains model, while (optionally) evaluating validation accuracy at the end of every epoch
    Args:
        model: Keras Model class to train
        params: dict of training params with keys: "learning_rate", "batch_size", "epochs",
        images (numpy array of images of dimensions (224, 224, 3)): training images
        labels (numpy array of integers 0/1): training labels
        validation_images: numpy array of validation images (optional)
        validation_labels: numpy array of validation labels (optional)
    Returns:
        train_images_amount: the amount of images used for training
        train_images_flower_amount: the amount of images tagged as flowers used for training
    """
    print("training model")
    model.fit(
        images,
        labels,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        validation_data=(validation_images, validation_labels) if validation_images else None,
    )
    train_images_amount = len(images)
    train_images_flower_amount = np.count_nonzero(labels)
    return train_images_amount, train_images_flower_amount

def test(images, model):
    """ Test model on images
        Args:
            images (numpy array of images of dimensions (224, 224, 3)): test images
            model (Model): Keras Model class
        Returns: error rate (double)
    """
    print("testing")
    probas = model.predict(images)
    return probas

def evaluate(probas, labels, start_time):
    """evaluate model performance
    Args:
        probas (numpy array of double): probability results from test
        labels (numpy array of int): numpy array of ground-truth test labels
        start_time (int): experiment start time
    Returns:
        precision (array of double): precision values for each threshold
        recall array of double): recall values for each threshold
        average_precision (double): average precision using scikit metrics
        error_rate (double): percentage of correct predictions for threshold 0.5
        conf_mat (2*2 Matrix): confusion matrix for threshold 0.5
        running_time (int): seconds elapsed since start_time
    """
    running_time = time.perf_counter() - start_time
    precision, recall, _ = precision_recall_curve(labels, probas)
    average_precision = average_precision_score(labels, probas)
    predictions =  (probas.flatten() > 0.5).astype("int")
    error_rate = get_error_rate(predictions, labels)
    conf_mat = confusion_matrix(labels, predictions, normalize='true')
    return precision, recall, average_precision, error_rate, conf_mat, running_time

def report(precision, recall, average_precision, error_rate, conf_mat, running_time, model, train_images_amount, train_flowers_amount):
    """Print & Plot results
    Args:
        Receives all returned values from evaluate(), and:
        model (Model): Keras Model class
        train_images_amount (int): number of training images used
        train_flowers_amount (int): number of training images labeled flowers
    """
    print("\nReport")
    print("\nNetwork Architecture")
    model.summary()
    print(f'\nTotal running time: {(running_time/60):0.4f} minutes')
    print(f'Flowers/Training Images: {train_flowers_amount}/{train_images_amount}')
    print('Error rate: {0:0.2f}'.format(error_rate))
    print(f'Average precision-recall score: {average_precision:0.2f}')
    print_confusion_matrix(conf_mat)
    plot_pr_curve(precision, recall, average_precision)

def print_confusion_matrix(conf_mat):
    """Print nicely formatted confusion matrix outputted from scitkit.metrics.confusion_matrix"""
    print("\nConfusion Matrix for threshold 0.5:")
    print(pd.DataFrame(
        conf_mat, 
        index=['true:flower', 'true:not-flower'], 
        columns=['pred:flower', 'pred:not-flower']
    ))

def plot_pr_curve(precision, recall, average_precision):
    """Plot Precision-Recall curve with average_precision
        Args:
            precision (array of double): precision values for each threshold
            recall (array of double): recall values for each threshold
            average_precision (double): average precision score
    """
    plt.plot(recall, precision)
    plt.fill_between(recall, precision, alpha=0.2)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(f'Precision Recall Curve for Binary Classification of Flower/Not Flower \n Average Precision: {average_precision:0.3f}')
    plt.show()

def get_error_rate(predictions, labels):
    """ Calculate error rate: (wrong predictions)/(total predictions)
        Args:
            predictions: numpy array of predictions
            labels: numpy array of ground-truth labels
        Returns: error rate (double)
    """
    accuracy = np.count_nonzero(predictions == labels)/len(labels)
    return 1 - accuracy

## used for creating graph for report
def plot_confusion_matrix(predictions, labels):
    skplt.metrics.plot_confusion_matrix(labels, predictions, normalize=True)

## for tuning
def create_scikit_model():
    """create a scikit keras wrapper so that we can use scikit k-fold and grid search for tuning """
    model = KerasClassifier(build_fn=createModel, epochs=params["model"]["epochs"], batch_size=32, verbose=0)
    return model

## for tuning
def kfold(model, images, labels):
    """ perform 3 fold training and evaluation, prints mean and std of results
        Args:
            model: instance of KerasClassifier (keras scikit wrapper)
            images (numpy array of images of dimensions (224, 224, 3)): test images
            labels: numpy array of ground-truth labels
    """
    kfold = StratifiedKFold(n_splits=3, shuffle=True)
    results = cross_val_score(model, images, labels, cv=kfold, verbose=1, n_jobs=-1)
    print("\nK-Folds results: \n")
    print("mean: ", results.mean())
    print("std: ", results.std())
    print()

## for tuning
def grid_search(model, images, labels):
    dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    param_grid = dict(dropout_rate=dropout)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose = 2, n_jobs=-1, refit=False)
    grid_result = grid.fit(images, labels) 

    # summarize tuning results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['meanz_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def get_worst_pred(probas, labels):
    """ Find the 5 worst false positives, and false negatives
        Args:
            probas: output from keras network
            labels: numpy array of ground-truth labels
        Returns: dictionary with keys "flower" and "not flower"
            each contains an array of 5 tuples, each tuple is an error, sorted by size of error in descending order
            each error is made up of the probability, and the index of the corresponding image.
    """
    print()
    class_names = ['flower', 'not flower']
    worst_pred = {i: [] for i in class_names}
    flower_idx = np.where(labels == 1)[0]
    not_flower_idx = np.where(labels == 0)[0]
    flower_probas = sorted(((probas[idx][0], idx) for idx in flower_idx), key=lambda x: x[0])
    not_flower_probas = sorted(((probas[idx][0], idx) for idx in not_flower_idx), key=lambda x: x[0], reverse=True)
    for i in range(5):
        flower_probas_value, flower_probas_index = flower_probas[i]
        if flower_probas_value < 0.5:
            worst_pred['flower'].append([flower_probas_index, flower_probas_value])
        not_flower_probas_value, not_flower_probas_index = not_flower_probas[i]
        if not_flower_probas_value > 0.5:
            worst_pred['not flower'].append([not_flower_probas_index, not_flower_probas_value])
    return worst_pred

main()
