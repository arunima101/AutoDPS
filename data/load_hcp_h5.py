import os
import glob
import numpy as np
import torch
import pathlib
from tqdm.notebook import tqdm
from monai.data import ITKReader,NibabelReader
from monai.transforms import LoadImage, LoadImaged
# from monai.transforms import (
#     Orientationd, AddChanneld, Compose, ToTensord, Spacingd,Resized,ScaleIntensityD,ResizeWithPadOrCropd
# )
from monai.data import Dataset
import h5py
import threading


def load_data(h5_path,mode):
#     data_path = data_path +  '/**'   
    
#     transforms_1 = Compose(
#     [
#      AddChanneld(('image')),
#      Orientationd(('image'),'RAS'),
#      Resized(keys = ('image'),spatial_size = (256, 256,-1),mode = 'trilinear' ,align_corners = True),
#      ScaleIntensityD(('image',)),
#      ToTensord(('image')),
#     ])
    h5 = h5_path + '/' + mode
#     print(f" Data Path: {h5}")
    transforms_1 = None
    dataset = LoadHCPh5(transforms_1,mode,h5cachedir=h5)

    return dataset

class LoadHCPh5(torch.utils.data.Dataset):
    
    def __init__(self,transforms,
                 mode='train',
                 nslices_per_image = 260 ,
                 start_slice = 110,
                 end_slice = 130,
                 h5cachedir=None):
        
        if h5cachedir is not None:
            if not os.path.exists(h5cachedir):
                os.mkdir(h5cachedir)
            self.cachedir = h5cachedir
            
        #### start_slice & end_slice---> slices to be truncated in each volume vol[:,:,start_slice:-end_slice]
        
        self.start_slice = start_slice
        self.end_slice = end_slice
        
        self.nslices = nslices_per_image - self.end_slice
            
        self.h5_list = []
        self.files = []
        for x in glob.glob(self.cachedir+'/**.h5'):
            self.h5_list.append(x)
            
        self.files = sorted([str(i) for i in self.h5_list])
        
#         self.examples = []
        
#         for fname in self.files:
#             self.examples += [(fname, slice) for slice in range(self.nslices)]


    def __len__(self):
        return len(self.files)*(self.nslices - self.start_slice)

    def __getitem__(self,index):
        
        data = {}
        
        ########### slicenum ##############
        slicenum = index % (self.nslices - self.start_slice)

        slicenum += self.start_slice
        #######################################
        
        filenum = index // (self.nslices - self.start_slice)
        h5name = self.files[filenum]
            
        vol = h5name.split("/")[-1].split(".")[0]

        with h5py.File(h5name,'r',libver='latest', swmr=True) as itm:
#                 print(f"{vol} , {slicenum}")

                for key in itm.keys(): 

                    data[key]=torch.from_numpy(itm[key][:,:,:,slicenum])
                        
        if len(data)>0:
                
            return data['image'],data['mask'],{"y":np.array((1))},vol,slicenum 
        
        else:
            # replace with random
            return self.__getitem__(np.random.randint(len(self.files)))