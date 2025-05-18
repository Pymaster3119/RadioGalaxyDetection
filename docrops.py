'''
Todo:
- Histogram the sizes
- Rotate by arbitrary angles
'''

import pickle
from collections import namedtuple
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
import pickle
from scipy import ndimage
import matplotlib.pyplot as plt
import tqdm

annotationclass = namedtuple('annotationclass', ['bbox','category','keypoints','segmentation'])
imageclass = namedtuple('imageclass', ['image', 'normalized', 'annotations', 'mask'])

with open("train.obj", 'rb') as f:
    train = pickle.load(f)

with open("test.obj", 'rb') as f:
    test = pickle.load(f)

# with open("val.obj", 'rb') as f:
#     val = pickle.load(f)

class AugmentedDataset():
    def __init__(self, data, transforms=False):
        self.data = data
        self.transforms = transforms
    def __len__(self):
        if self.transforms:
            return len(self.data) * 10
        else:
            return len(self.data)

    def getnumblobs(self, idx):
        idx = idx % len(self.data)
        sample = self.data[idx]
        X = sample.normalized.astype(np.float32)
        Y = sample.mask.astype(np.float32)
        
        # Randomly select one of the blobs in Y
        blob_channels = np.where(Y.sum(axis=(0, 1)) > 0)[0]
        return blob_channels
    def getitem(self, idx, blobchannel):
        idx = idx % len(self.data)
        sample = self.data[idx]
        X = sample.normalized.astype(np.float32)
        Y = sample.mask.astype(np.float32)
        
        # Randomly select one of the blobs in Y
        blob_channels = np.where(Y.sum(axis=(0, 1)) > 0)[0]
        selected_channel = blobchannel
        
        # Find the coordinates of the selected blob
        blob_coords = np.argwhere(Y[:, :, selected_channel] > 0)
        selected_blob_coord = random.choice(blob_coords)

        # Crop out the entire blob
        blob_mask = Y[:, :, selected_channel] > 0
        blob_coords = np.argwhere(blob_mask)
        if len(blob_coords) == 0:
            return None#self.__getitem__((idx + 1) % len(self.data))
        x_min, y_min = blob_coords.min(axis=0)
        x_max, y_max = blob_coords.max(axis=0)
        blob = X[x_min:x_max + 1, y_min:y_max + 1, :]
        # Calculate the scaling factor to fit the blob within 64x64 while maintaining aspect ratio
        scale = min(64 / blob.shape[0], 64 / blob.shape[1])
        if scale > 4:
            return None
        new_shape = (int(blob.shape[0] * scale), int(blob.shape[1] * scale), blob.shape[2])
        
        # Resize the blob
        resized_blob = ndimage.zoom(blob, (new_shape[0] / blob.shape[0], new_shape[1] / blob.shape[1], 1), order=1)
        
        # Create a 64x64xC array filled with zeros (padding)
        padded_blob = np.zeros((64, 64, blob.shape[2]), dtype=np.float32)
        
        # Calculate the top-left corner to center the resized blob
        x_offset = (64 - resized_blob.shape[0]) // 2
        y_offset = (64 - resized_blob.shape[1]) // 2
        
        # Place the resized blob in the center of the padded array
        padded_blob[x_offset:x_offset + resized_blob.shape[0], y_offset:y_offset + resized_blob.shape[1], :] = resized_blob
        
        blob = padded_blob
        
        X_cropped = blob
        # Add noise to the cropped image
        if self.transforms:
            noise = np.random.normal(0, 0.1, X_cropped.shape).astype(np.float32)
            X_cropped = noise + X_cropped
            X_cropped = np.clip(X_cropped, 0, 1)
        return X_cropped, selected_channel

train_dataset = AugmentedDataset(train)
test_dataset = AugmentedDataset(test)
#val_dataset = AugmentedDataset(val)

#Save train set
X = []
y = []
totalblobs = 0
failiures = 0
for i in tqdm.tqdm(range((len(train_dataset)))):
    blobs = train_dataset.getnumblobs(i)
    for blob in blobs:
        try:
            totalblobs += 1
            X_cropped, selected_channel = train_dataset.getitem(i, blob)
            if X_cropped is not None:
                X.append(X_cropped)
                y.append(selected_channel)
            else:
                failiures += 1
        except Exception as e:
            failiures += 1
            continue

print(f"Total blobs: {totalblobs}")
print(f"Failures: {failiures}")

with open("train_cropped_X.obj", 'wb') as f:
    pickle.dump(X, f)
with open("train_cropped_y.obj", 'wb') as f:
    pickle.dump(y, f)

#Save test set
X = []
y = []
totalblobs = 0
failiures = 0
for i in tqdm.tqdm(range((len(test_dataset)))):
    blobs = test_dataset.getnumblobs(i)
    for blob in blobs:
        try:
            totalblobs += 1
            X_cropped, selected_channel = test_dataset.getitem(i, blob)
            if X_cropped is not None:
                X.append(X_cropped)
                y.append(selected_channel)
            else:
                failiures += 1
        except Exception as e:
            failiures += 1
            print(e)
            continue

print(f"Total blobs: {totalblobs}")
print(f"Failures: {failiures}")

with open("test_cropped_X.obj", 'wb') as f:
    pickle.dump(X, f)
with open("test_cropped_y.obj", 'wb') as f:
    pickle.dump(y, f)

# #Save val set
# X = []
# y = []
# for i in range(len(val_dataset)):
#     blobs = val_dataset.getnumblobs(i)
#     for blob in blobs:
#         try:
#             X_cropped, selected_channel = val_dataset.getitem(i, blob)
#             if X_cropped is not None:
#                 X.append(X_cropped)
#                 y.append(selected_channel)
#         except Exception as e:
#             print(e)
#             continue
# with open("val_cropped_X.obj", 'wb') as f:
#     pickle.dump(X, f)
# with open("val_cropped_y.obj", 'wb') as f:
#     pickle.dump(y, f)