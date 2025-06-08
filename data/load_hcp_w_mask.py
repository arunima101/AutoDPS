import os
import glob
import numpy as np
import torch
from tqdm.notebook import tqdm
from monai.data import ITKReader,NibabelReader
from monai.transforms import LoadImage, LoadImaged
from monai.transforms import (
    Orientationd, AddChanneld, Compose, ToTensord, Spacingd,Resized,ScaleIntensityD,ResizeWithPadOrCropd,Rotate90
)
from monai.data import Dataset
import h5py
import threading


def load_data(data_path,h5_path,mode):
    data_path = data_path +  '/**'   
    
    transforms_1 = Compose(
    [
     AddChanneld(('image')),
     Orientationd(('image'),'RAS'),
     Resized(keys = ('image'),spatial_size = (256, 256,-1),mode = 'trilinear' ,align_corners = True),
     ScaleIntensityD(('image',)),
     ToTensord(('image')),
    ])
    h5_path = h5_path + '/' + mode
    
    dataset = H5CachedDataset(data_path,transforms_1,mode = mode,h5cachedir=h5_path)

    return dataset


# def normalize(img):
#     return (img -img.min())/(img.max()-img.min())


class H5CachedDataset(Dataset):
    def __init__(self, datapath,
                 transforms_1, 
                 nslices_per_image = 260 ,
                 start_slice = 110,
                 end_slice = 130,
                 mode = "train",
                 h5cachedir=None):

        if h5cachedir is not None:
            if not os.path.exists(h5cachedir):
                os.mkdir(h5cachedir)
            self.cachedir = h5cachedir
        niilist=[]
        masklist = []
        for x in glob.glob(datapath):
            niilist.append(x +'/MNINonLinear/T2w.nii.gz')
        
        for x in glob.glob(datapath):
            masklist.append(x + '/T1w/brainmask_fs.nii.gz')
            
        self.train_datalist = niilist[50:300]  #### training on 250 vol, 20 slices 110-130
        self.test_datalist = niilist[:50]      #### testing on 40 volumes
        if mode=='train':
            self.datalist = self.train_datalist
            self.mask_list = masklist[50:300]
        else:
            self.datalist = self.test_datalist
            self.mask_list = masklist[:50]
            
        self.xfms = transforms_1
                    
        #### 3d image loader from monai
        
        self.loader = LoadImage()
        self.loader.register(NibabelReader())
        
        #### start_slice & end_slice---> slices to be truncated in each volume vol[:,:,start_slice:-end_slice]
        
        self.start_slice = start_slice
        self.end_slice = end_slice
        
        self.nslices = nslices_per_image - self.end_slice

        
    def __len__(self):
        #### total number of slices in all the volumes
        return len(self.datalist)*(self.nslices - self.start_slice)
    
    def clear_cache(self):
        #### function to clear the directory storing h5 files (used for caching the h5 files)
        for fn in os.listdir(self.cachedir):
            os.remove(self.cachedir+'/'+fn)
            
    def __getitem__(self,index):
        #### ditionary to store data slicewise
        data = {}
        label ={}
#      
#        
        filenum = index // (self.nslices - self.start_slice)


        slicenum = index % (self.nslices - self.start_slice)

        slicenum += self.start_slice

#         print(f"\n VOLUME {filenum} SLICE {slicenum}")

        #### Extract the datafile location & mask file location based on filenum

        datalist_filenum = self.datalist[filenum]
        masklist_filenum = self.mask_list[filenum]
#         print(datalist_filenum)
        loc_data = datalist_filenum
        
        
        
            
        ##### if h5 exists for the current volume fill data dictionary with current slice number
        if self.cachedir is not None:
            h5name = self.cachedir+'/%d.h5' % filenum

            ptname = self.cachedir+'/%d.pt' % filenum
            print(f"\n H5 {h5name}\n slice {slicenum}")
            if os.path.exists(h5name):
                
                with h5py.File(h5name,'r',libver='latest', swmr=True) as itm:
                    for key in itm.keys():                       
                        data[key]=torch.from_numpy(itm[key][:,:,:,slicenum])



        ##### if data dictionary is empty #######
        if len(data)==0:
            
            
            ######### Loading data #########################
            imgdata, meta = self.loader(loc_data)
            imgdata_padded = np.zeros((320,320,260))
            imgdata_padded[30:-30,5:-4,:] = imgdata

            ### Rotating it ###
            torch_data = torch.from_numpy(imgdata_padded)
            imgdata = torch.rot90(torch_data,1)
            
            ################## Loading mask ###################
            maskvolume,meta = self.loader(masklist_filenum)
            maskdata_padded = np.zeros((320,320,260))
            maskdata_padded[30:-30,5:-4,:]=maskvolume ### making it 320x320

            ### Rotating it ###
            torch_mask_data = torch.from_numpy(maskdata_padded)
            maskdata = torch.rot90(torch_mask_data,1)

            mask3d = self.xfms({'image':maskdata})
        
            ###################################################

            #### store volume wise image in a dictionary 
            data_i = {'image':imgdata,'mask':mask3d['image']}

            #### transform the data dictionary
            data3d = self.xfms(data_i)
            

            if self.cachedir is not None:
                other = {}

                with h5py.File(h5name,'w',libver='latest') as itm:
                    itm.swmr_mode = True
                    for key in data3d:
                        if key in ['image','mask']:                             
                            img_npy = data3d[key].numpy()

                            shp = img_npy.shape
                            
                            chunk_size = list(shp[:-1])+[1]
                            ds = itm.create_dataset(key,shp,chunks=tuple(chunk_size),dtype=img_npy.dtype)
                            ds[:]=img_npy[:]
                            ds.flush()
                    else:
                        other[key]=data3d[key]
                torch.save(other,ptname)
                
            
            data = {'image':data3d['image'][:,:,:,slicenum],'mask':data3d['mask'][:,:,:,slicenum]}

            
        if len(data)>0:
#             print(f"\n VOLUME {filenum} SLICE {slicenum}")
            res = {}
            res['image'] = data['image'] 
            res['mask'] = data['mask']
            res['filenum'] = filenum
            res['slicenum'] = slicenum
            res['idx'] = index
            return res['image'] ,res['mask'], {"y":np.array((1))} 

        else:
            # replace with random
            return self.__getitem__(np.random.randint(len(self.datalist)))
