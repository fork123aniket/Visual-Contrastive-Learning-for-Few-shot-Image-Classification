import numpy as np
import pickle
import os
import cv2
from PIL import Image

train_folder = "data/images_background/"
val_folder = 'data/images_evaluation/'
save_path = 'data/'


def loadimgs(path, n=0):
    X = []
    y = []
    cat_dict = {}
    lang_dict = {}
    curr_y = n
    for alphabet in os.listdir(path):
        print("loading alphabet: " + alphabet)
        lang_dict[alphabet] = [curr_y, None]
        alphabet_path = os.path.join(path, alphabet)
        for letter in os.listdir(alphabet_path):
            cat_dict[curr_y] = (alphabet, letter)
            category_images = []
            letter_path = os.path.join(alphabet_path, letter)
            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                image = Image.open(image_path)
                image = image.convert("L")
                # image = cv2.imread(image_path, 0)
                category_images.append(image)
                y.append(curr_y)
            try:
                X.append(np.stack(category_images))
            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)
            curr_y += 1
            lang_dict[alphabet][1] = curr_y - 1
    y = np.vstack(y)
    X = np.stack(X)
    return X, y, lang_dict


X, y, c = loadimgs(train_folder)

with open(os.path.join(save_path, "train.pickle"), "wb") as f:
    pickle.dump((X, c), f)

Xval, yval, cval = loadimgs(val_folder)

with open(os.path.join(save_path, "val.pickle"), "wb") as f:
    pickle.dump((Xval, cval), f)
