from sklearn.decomposition import PCA
import numpy as np
import cv2
from scipy import ndimage


def pca(data):
    pca = PCA(n_components=50)

    pca.fit(data)
    X = pca.transform(data)
    return X


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape)[:2]/2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale=1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[:2],flags=cv2.INTER_LINEAR)
    return result


def apply_rotation(data, label):
    new_data = []
    new_label = []
    for i in range(len(data)):
        if i % 3 == 0:
            new_data.append(rotate_image(data[i], 5))
        elif i % 3 == 1:
            new_data.append(rotate_image(data[i], 15))
        elif i % 3 == 2:
            new_data.append(rotate_image(data[i], 10))

        new_label.append(label[i])

    return new_data, new_label


def shift_image(image, shift):
    new_img_ = ndimage.shift(image, shift, cval=0)
    return new_img_


def apply_shifting(data, label):
    new_data = []
    new_label = []
    for i in range(len(data)):
        new_data.append(shift_image(data[i], 50))
        new_label.append(label[i])

    return new_data, new_label


def contrast_image(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output


def apply_contrast(data, label):
    new_data = []
    new_label = []
    for i in range(len(data)):
        new_data.append(contrast_image(data[i]))
        new_label.append(label[i])

    return new_data, new_label


def vertical_flip_image(image):
    img = cv2.flip(image, 1)

    return img


def apply_vertical_flip(data, label):
    new_data = []
    new_label = []
    for i in range(len(data)):
        new_data.append(vertical_flip_image(data[i]))
        new_label.append(label[i])

    return new_data, new_label