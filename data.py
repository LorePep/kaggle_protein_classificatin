import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
import cv2

from itertools import chain
from collections import Counter

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


class data_generator:
    
    def create_train(dataset_info, batch_size, shape, augument=True):
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))

            for i, idx in enumerate(random_indexes):
                image = data_generator.load_image(
                    dataset_info.iloc[idx]['path'], shape)   
                if augument:
                    image = data_generator.augment(image)
                batch_images[i] = image
                batch_labels[i][dataset_info.iloc[idx]['target_list']] = 1
            yield batch_images, batch_labels
            
    
    def load_image(path, shape):
        colors = ["red","green","blue","yellow"]
        flags = cv2.IMREAD_GRAYSCALE
        imgs = []

        for color in colors:
            img = cv2.imread(path+'_'+color+'.png', flags).astype(np.float32)/255
            img = cv2.resize(img, (shape[0], shape[1]), cv2.INTER_AREA)
            imgs.append(img)  
        return np.stack(imgs, axis=-1)
            
    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=30),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug


def get_data(image_folder, csv_path):
    image_df = pd.read_csv(os.path.join(csv_path))
    print(image_df.shape[0], 'masks found')
    print(image_df['Id'].value_counts().shape[0])
    # just use green for now

    image_df['path'] = image_df['Id'].map(lambda x: os.path.join(image_folder, x))
    image_df['target_list'] = image_df['Target'].map(lambda x: [int(a) for a in x.split(' ')])
    
    all_labels = list(chain.from_iterable(image_df['target_list'].values))
    c_val = Counter(all_labels)
    n_keys = c_val.keys()
    max_idx = max(n_keys)

    image_df['target_vec'] = image_df['target_list'].map(lambda ck: [i in ck for i in range(max_idx+1)])

    return image_df

def split_data(data):
    raw_train_df, valid_df = train_test_split(data, 
                 test_size = 0.3, 
                  # hack to make stratification work                  
                 stratify = data['Target'].map(lambda x: x[:3] if '27' not in x else '0'))
    print(raw_train_df.shape[0], 'training masks')
    print(valid_df.shape[0], 'validation masks')

    return raw_train_df, valid_df
