"""
This script loads the CIFAR-10 dataset and trains classifiers on it.

Author: Laura Bock Paulsen (202005791)
"""
# data loader
from tensorflow.keras.datasets import cifar10

import cv2
import argparse
import numpy as np
from img_clf import ImageClassifier
import os


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--clf", type=str, default = "logistic", help = "type of classification model to use")
    args = vars(ap.parse_args())

    # check if model is valid
    if args["clf"] not in ["logistic", "mlp"]:
        raise ValueError("Only 'logistic' and 'mlp' are implemented for now.")
    
    return args

def preprocess_image(img:np.ndarray):
    """
    prepares an image for classification (converts to gray-scale, scales to values between 0 and 1, and reshapes to 1D array)

    Parameters
    ----------
    img : numpy.ndarray
        image to be preprocessed.

    Returns
    -------
    img : numpy.ndarray
        preprocessed image.
    """

    # convert to gray-scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # scaling to 0-1 by dividing 
    img = img / 255.0

    # reshape to 1 dimensional array
    img = img.reshape((img.shape[0] * img.shape[1]))

    return img

def main():
    # parse arguments
    args = parse_args()

    # load data 
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # define labels for the classes (used in for classification report)
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
    # preprocess images in both test and training set
    X_train = np.array([preprocess_image(image) for image in X_train])
    X_test = np.array([preprocess_image(image) for image in X_test])

    # initialize classifier
    iclf = ImageClassifier(model_type = args["clf"])

    # train classifier
    iclf.train(X_train, y_train)

    # predict on test set
    y_pred = iclf.predict(X_test)

    # classification report
    report = iclf.evaluate(y_test, y_pred, target_names=labels)

    # save report to file
    with open(os.path.join("out", f"{args['clf']}_clf_report.txt"), "w") as f:
        f.write(report)


if __name__ in "__main__":
    main()
