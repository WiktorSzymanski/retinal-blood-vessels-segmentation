import pickle
import cv2
import os
import shutil
import skimage
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import copysign, log10
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, binary_dilation, binary_erosion

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.metrics import classification_report_imbalanced


import xgboost as xgb

from sklearn.utils import shuffle
from imblearn.under_sampling import RandomUnderSampler


def prep_tif_file(file):
    _, thresholded = cv2.threshold(file, 3, 255, cv2.THRESH_BINARY)
    return thresholded


def normalize_image(image):
    image = np.array(image)

    max = np.max(image)
    min = np.min(image)

    div = max - min
    return (image - min)/div


def preprocess_image(image):
    green = image[:,:,1]
    blured = cv2.GaussianBlur(green, (5, 5), 0)
    weighted = cv2.addWeighted(green, 2, blured, 1, 10)

    return weighted


def get_mask(image):
    image = cv2.blur(image, (5, 5), 0)
    _, mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    return mask


def get_vessels_im_pro(image):
    preprocessed = preprocess_image(image)
    franged = frangi(preprocessed)
    normalized = normalize_image(franged)*255
    _, thresholded = cv2.threshold(normalized, 3, 255, cv2.THRESH_BINARY)

    filtered = remove_small_objects(thresholded > 0, min_size=4000)
    mask = get_mask(image)

    vessels = filtered * mask[:,:,0]

    return vessels


def show_images(images):
    n = len(images)
    plt.figure(figsize=(16, 5 * n))
    for i, image in enumerate(images):
        plt.subplot(n, 3, 3 * i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image[0], cmap="gray")

        plt.subplot(n, 3, 3 * i + 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image[1], cmap="gray")

        plt.subplot(n, 3, 3 * i + 3)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image[2], cmap="gray")


def calc_metrics(y_true, y_pred):
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()

    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_pred, y_true),
        display_labels=[False, True]).plot()
    plt.show()
    print(classification_report_imbalanced(y_true, y_pred))


def prepere_image_data(image, mask, size):
    if (size % 2 == 0):
        raise Exception("Size must be odd")
    
    print(mask.shape)

    features = []
    labels = []

    for i in range(0, image.shape[0] - 1, size):
        for j in range(0, image.shape[1] - 1, size):
            part_image = image[i:i + size, j:j + size]
            part_mask = mask[i:i + size, j:j + size]

            # if part_mask.sum() == 0 or part_image.shape != (size, size, 3):
            #     continue

            red_mean = part_image[:,:,0].mean()
            green_mean = part_image[:,:,1].mean()
            blue_mean = part_image[:,:,2].mean()

            red_var = part_image[:,:,0].var()
            green_var = part_image[:,:,1].var()
            blue_var = part_image[:,:,2].var()

            hu_moments = get_hu_moments(part_image[:,:,1]).flatten()

            features.append([red_mean, green_mean, blue_mean, red_var, green_var, blue_var, *hu_moments])
            labels.append(1 if part_mask[size//2][size//2] == 255 else 0)

    return features, labels


def prepere_multiple_images(data, size, dir="data", file_name="dataset", save_to_csv=True):
    
    all_features = list()
    all_labels = list()

    for image, mask in data:
        features, labels = prepere_image_data(image, mask, size)

        all_features.extend(features)
        all_labels.extend(labels)

    df = pd.DataFrame(all_features, columns=["red_mean", "green_mean", "blue_mean",
                                             "red_var", "green_var", "blue_var",
                                             "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7"])

    df["label"] = all_labels

    if not save_to_csv:
        return df
    
    prepere_dir(dir)
    df.to_csv(f"{dir}/{file_name}.csv", index=False)
        



def get_hu_moments(image):
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments)

    for i in range(0,7):
        if hu_moments[i] != 0:
            hu_moments[i] = -1 * copysign(1.0, hu_moments[i]) * log10(abs(hu_moments[i]))

    return hu_moments[:7]


def prepere_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def clear_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.mkdir(dir)


def prep_classifier(train, test):
    X_train = train[train.columns.difference(['label'])]
    y_train = train['label']
    rus = RandomUnderSampler(sampling_strategy=0.6)
    X_train, y_train = rus.fit_resample(X_train, y_train)

    X_test = test[test.columns.difference(['label'])]
    y_test = test['label']

    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    calc_metrics(y_test, pred)

    return clf


def pred_image(clf, image, mask):
    width, height, _ = image.shape

    df = prepere_multiple_images([(image, mask)], 5, save_to_csv=False)

    y_pred = clf.predict(df[df.columns.difference(['label'])])


    mask = cv2.resize(mask, dsize=(height//5 + 1, width//5), interpolation=cv2.INTER_NEAREST)
    mask = [True if i > 0 else False for i in mask.flatten()]

    pred_img = np.reshape(y_pred, (width//5, height//5 + 1))

    normalized = normalize_image(pred_img)*255
    filtered = remove_small_objects(normalized > 0, min_size=0)

    return filtered, mask


def save_model(clf, dir="models", file_name="model"):
    prepere_dir(dir)
    pickle.dump(clf, open(f"{dir}/{file_name}.pickle.dat", "wb"))


def load_model(dir="models", file_name="model"):
    return pickle.load(open(f"{dir}/{file_name}.pickle.dat", "rb"))

