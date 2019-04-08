# -*- coding: utf-8 -*-
#
# Natural image datset by Bruno Olshausen 
# http://www.rctn.org/bruno/sparsenet/
import numpy as np
from scipy import io


class Dataset(object):
    def __init__(self):
        file_path = "./data/IMAGES.mat"
        matdata = io.loadmat(file_path)
        images = matdata['IMAGES'].astype(np.float32)
        patches = []

        image_size = images.shape[2]
        
        # (512, 512, 10)
        for image_index in range(image_size):
            for i in range(0, 512, 16):
                for j in range(0, 512, 16):
                    patch = images[i:i+16, j:j+16, image_index]
                    patches.append(patch)
        
        patches = np.array(patches, dtype=np.float32)
        patches = np.reshape(patches, [-1, 256])
        self.patches = patches

    def __getitem__(self, index):
        return self.patches[index]

    def __len__(self):
        return len(self.patches)

