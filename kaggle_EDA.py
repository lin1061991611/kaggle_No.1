import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pydicom
import glob
from tqdm.notebook import tqdm
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.pyplot as plt
from skimage import exposure
import cv2
import warnings
from fastai.vision.all import *
from fastai.medical.imaging import *

train_data_path = Path('/media/lc/本地磁盘/kaggle_train_data')
train_label_path = Path('/media/lc/本地磁盘/kaggle_label')

study_classes = ['Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']
train_study_df = pd.read_csv(train_label_path / 'train_study_level.csv')
train_image_df = pd.read_csv(train_label_path / 'train_image_level.csv')

def dicom2array(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.read_file(path)
    # VOI LUT (if available by DICOM device) is used to
    # transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

def show_class_frequency(train_study_df, study_classes):
    plt.figure(figsize=(10, 5))
    class_frequency_sum = train_study_df[study_classes].values.sum(axis=0)
    print(class_frequency_sum)
    plt.bar([1, 2, 3, 4], class_frequency_sum)
    plt.xticks([1, 2, 3, 4], study_classes)
    plt.ylabel('Frequency')
    plt.show()

def remove_nolabel(train_image_df):
    new_train_image_df = train_image_df.copy()

def plot_imgs(imgs, cols=4, size=7, is_rgb=True, title="", cmap='gray', img_size=(600,600)):
    rows = len(imgs)//cols + 1
    fig = plt.figure(figsize=(cols*size, rows*size))
    for i, img in enumerate(imgs):
        if img_size is not None:
            img = cv2.resize(img, img_size)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()


def show_random_bbox(train_image_df):
    #train_image_df['image_path'] = train_image_df.apply(image_path, axis=1)
    imgs = []
    dicom_paths = get_dicom_files(train_data_path)
    # map label_id to specify color
    thickness = 10
    scale = 5
    for i in range(8):
        image_path = random.choice(dicom_paths)
        print(image_path)
        img = dicom2array(path=image_path)
        img = cv2.resize(img, None, fx=1 / scale, fy=1 / scale)
        img = np.stack([img, img, img], axis=-1)
        for i in train_image_df.loc[obj['image_path'] == image_path].split_label.values[0]:
            if i[0] == 'opacity':
                img = cv2.rectangle(img,
                                    (int(float(i[2]) / scale), int(float(i[3]) / scale)),
                                    (int(float(i[4]) / scale), int(float(i[5]) / scale)),
                                    [255, 0, 0], thickness)

        img = cv2.resize(img, (600, 600))
        imgs.append(img)

    plot_imgs(imgs, cmap=None)


if __name__ == "__main__":
    #show_class_frequency(train_label_path, study_classes)
    #remove_nolabel(train_image_df)
    show_random_bbox(train_image_df)

