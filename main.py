#
#   OSMAN FURKAN KINLI
#   S002969 - Computer Science in Engineering
#   Ozyegin University - Senior Design Project Part 2
#   Classification of Vehicles on Highway
#

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pandas as pd
import cv2

import data_manipulation
import classifier
import os


def read_data():
    images = []
    for filename in os.listdir('C:/Users/Furkan/Desktop/Bitirme/dataset/images'):
        img = cv2.imread(os.path.join('C:/Users/Furkan/Desktop/Bitirme/dataset/images',filename))
        if img is not None:
            # For labelling manually.
            # cv2.imshow("Frame", img)
            # cv2.waitKey(0)
            images.append(img)

    return images


def reshape_list(list):
    new_list = []
    for img in list:
        img = img.reshape(4800)
        new_list.append(img)

    return new_list


def read_labels():
    data = pd.read_csv('C:/Users/Furkan/Desktop/Bitirme/dataset/images/Labels.csv')
    X = data.iloc[0:, 1].values
    label_list = X

    return label_list


def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test


def main():
    ######
    # mat = get_data.read_data("C:/Users/Furkan/Desktop/Bitirme/dataset/video4.mp4")
    ######
    print("Dataset has been reading...")
    data = read_data()
    labels = read_labels()
    print("Dataset has been read.")

    print('Applying contrast streching manipulation...')
    contrasted_data, contrasted_labels = data_manipulation.apply_contrast(data, labels)
    print('Size of contrasted data: %d' % len(contrasted_data))

    print('Applying rotation manipulation...')
    rotated_data, rotated_labels = data_manipulation.apply_rotation(contrasted_data, labels)
    print('Size of rotated data: %d' % len(rotated_data))

    print('Applying shifting manipulation...')
    shifted_data, shifted_labels = data_manipulation.apply_shifting(contrasted_data, labels)
    print('Size of shifted data: %d' % len(shifted_data))

    print('Applying flipping manipulation...')
    flipped_data, flipped_labels = data_manipulation.apply_horizontal_flip(contrasted_data, labels)
    print('Size of shifted data: %d' % len(flipped_data))

    print('Concatenating manipulated data')
    concat_data = rotated_data + shifted_data + contrasted_data + flipped_data
    # concat_data = data  data For 2.5k sized Original data.

    print("Reshaping images...")
    reshaped_concat_data = reshape_list(concat_data)
    print('Shape of data: %s' % str(reshaped_concat_data[0].shape))

    print("PCA has been applying...")
    data_pca = data_manipulation.pca(reshaped_concat_data)
    print("PCA has been applied.")

    data = data_pca
    concat_labels = labels

    print("Spliting dataset into training and test set...")
    X_train, X_test, y_train, y_test = split_dataset(data, concat_labels[:len(data)])

    start = datetime.now()
    print("Appyling K-Nearest Neighbours Classifier...")
    knn_labels = classifier.knn_classifier(X_train, y_train, X_test)
    print("Evaluating accuracy...")
    classifier.evaluate(y_test, knn_labels)
    cm_knn = confusion_matrix(y_test, knn_labels)
    print("Confusion matrix: %s \n\t" % str(cm_knn))
    print("Running time: %s" % str(datetime.now() - start))

    start = datetime.now()
    print("Appyling Support Vector Machines Classifier...")
    svm_labels = classifier.svm_classifier(X_train, y_train, X_test)
    print("Evaluating accuracy...")
    classifier.evaluate(y_test, svm_labels)
    cm_svm = confusion_matrix(y_test, svm_labels)
    print("Confusion matrix: %s \n\t" % str(cm_svm))
    print("Running time: %s" % str(datetime.now() - start))

    start = datetime.now()
    print("Appyling Naive Bayes Classifier...")
    nbc_labels = classifier.naive_bayes_classifier(X_train, y_train, X_test)
    print("Evaluating accuracy...")
    classifier.evaluate(y_test, nbc_labels)
    cm_nbc = confusion_matrix(y_test, nbc_labels)
    print("Confusion matrix: %s \n\t" % str(cm_nbc))
    print("Running time: %s" % str(datetime.now()-start))

    start = datetime.now()
    print("Appyling Decision Tree Classifier...")
    dtc_labels = classifier.decision_tree_classifier(X_train, y_train, X_test)
    print("Evaluating accuracy...")
    classifier.evaluate(y_test, dtc_labels)
    cm_dtc = confusion_matrix(y_test, dtc_labels)
    print("Confusion matrix: %s \n\t" % str(cm_dtc))
    print("Running time: %s" % str(datetime.now() - start))

    start = datetime.now()
    print("Appyling Random Forest Classifier...")
    rfc_labels = classifier.random_forest_classifier(X_train, y_train, X_test)
    print("Evaluating accuracy...")
    classifier.evaluate(y_test, rfc_labels)
    cm_rfc = confusion_matrix(y_test, rfc_labels)
    print("Confusion matrix: %s \n\t" % str(cm_rfc))
    print("Running time: %s" % str(datetime.now() - start))

    """print("Applying K-Means Clustering...")
    classifier.kmeans_clustering(x_train=X_train, y_train=y_train)"""


if __name__ == '__main__':
    main()
