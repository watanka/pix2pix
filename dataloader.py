import torch.nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from glob import glob
import os
import matplotlib.pyplot as plt

class MapDataset(Dataset) :
    
    def __init__(self, data_dir, transform = None, train = True) :
        self.data_dir = data_dir
        self.img_list = glob(os.path.join(self.data_dir, '*.jpg'))
        self.transform = transform
        self.train = train
        
    def load_img(self, fname) :
        img = plt.imread(fname)
        real_img = img[:,:600]
        map_img = img[:,600:]
        return (real_img, map_img)
        
    def __getitem__(self, index) :
        
        if self.train :
            real_img, map_img = self.load_img(self.img_list[index])

            if self.transform :
                real_img = self.transform(real_img)
                map_img = self.transform(map_img)

            return (real_img, map_img)
        else : 
            img = plt.imread(self.img_list[index])
            if self.transform :
                img = self.transform(img)
            
            return img
            
        
    def __len__(self) :
        return len(self.img_list)