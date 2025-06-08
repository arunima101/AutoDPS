import os
import glob
import numpy as np
import torch
from tqdm.notebook import tqdm
import nibabel as nib
from monai.data import ITKReader,NibabelReader
from monai.transforms import LoadImage, LoadImaged
# from monai.transforms import (
#     Orientationd, AddChanneld, Compose, ToTensord, Spacingd,Resized,ScaleIntensityD,ResizeWithPadOrCropd,Rotate90
# )
from monai.data import Dataset
import h5py
import threading

def normalize(image):
    return (image - image.min())/(image.max()-image.min())

def load_data(data_path,h5_path,mode):
#     data_path = data_path +  '/**'   
    
#     transforms_1 = Compose(
#     [
#      AddChanneld(('image')),
#      Orientationd(('image'),'RAS'),
#      Resized(keys = ('image'),spatial_size = (256, 256,-1),mode = 'trilinear' ,align_corners = True),
#      ScaleIntensityD(('image',)),
#      ToTensord(('image')),
#     ])
    
    h5_path = h5_path + '/' + mode
    
    dataset = H5CachedDataset(data_path,mode = mode,h5cachedir=h5_path)

    return dataset



class H5CachedDataset(Dataset):
    def __init__(self, datapath,
                 nslices_per_image = 256 ,
                 start_slice = 155,
                 end_slice = 85,
                 mode = "train",
                 h5cachedir=None):

        if h5cachedir is not None:
            if not os.path.exists(h5cachedir):
                os.makedirs(h5cachedir)
            self.cachedir = h5cachedir
            
        niilist=[]
        hm1list = []
        hm2list = []
        masklist = []
        missing_data = ['sub-105822','sub-136038','sub-185823','sub-279373','sub-424015','sub-598630','sub-613957','sub-694330']
        
        for x in glob.glob(datapath + '/**/anat/**standard_T1w.nii.gz'):
            
            basepath = x.split("/")[:-1]
            fileid = x.split("/")[-3]
            if fileid in missing_data:
                continue
            niilist.append(x)
            filetype = x.split("/")[-1]
            hm1file = [fileid+ '_acq-headmotion1_T1w.nii.gz']
            hm2file = [fileid+ '_acq-headmotion2_T1w.nii.gz']
            maskfile = [fileid+'_mask.nii.gz']
            hm1path = "/".join(basepath+hm1file)
            hm2path = "/".join(basepath+hm2file)
            maskpath = "/".join(basepath+maskfile)
            hm1list.append(hm1path)
            hm2list.append(hm2path)
            masklist.append(maskpath)
#         for x in glob.glob(datapath + '/**/anat/**mask.nii.gz'):
#             masklist.append(x)
            
        
            
        self.train_datalist = niilist[:100]  #### 
        self.test_datalist = niilist[100:]
#         if motion_level ==1:
#             self.test_datalist = hm1list[108:]      
#         else:
#             self.test_datalist = hm2list[108:]
             
#         print(f"train_datalist {self.train_datalist}")
        if mode=='train':
            self.datalist = self.train_datalist
            self.hm1_datalist = hm1list[:100]
            self.hm2_datalist = hm2list[:100]
            self.mask_list = masklist[:100]
        else:
            self.datalist = self.test_datalist
            self.hm1_datalist = hm1list[100:]
            self.hm2_datalist = hm2list[100:]            
            self.mask_list = masklist[100:]
            
#         self.xfms = transforms_1
                    
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
        hm1_filenum = self.hm1_datalist[filenum]
        hm2_filenum = self.hm2_datalist[filenum]
        
            
        ##### if h5 exists for the current volume fill data dictionary with current slice number filenum {filenum} 
        if self.cachedir is not None:
            h5name = self.cachedir+'/%d.h5' % filenum
            vol = int(h5name.split('/')[-1].split('.')[0])
            ptname = self.cachedir+'/%d.pt' % filenum

            if os.path.exists(h5name):
                
                with h5py.File(h5name,'r',libver='latest', swmr=True) as itm:
                    for key in itm.keys():                       
                        data[key]=torch.from_numpy(itm[key][:,:,slicenum])


        
        ##### if data dictionary is empty #######self.loader(datalist_filenum)
        if len(data)==0:
            
            
            ######### Loading data #########################
            imgdata = nib.load(datalist_filenum)
            imgdata = imgdata.get_fdata()
            imgdata_padded = np.zeros((256,256,256))
            imgdata_padded[32:224,:,:] = imgdata
            torch_data = torch.from_numpy(imgdata_padded)
            imgdata = torch.rot90(torch_data,1)
            
            #### Headmotion 1 ###
            hm1data, meta = self.loader(hm1_filenum)
            hm1data_padded = np.zeros((256,256,256))
            hm1data_padded[32:224,:,:] = hm1data
            torch_data = torch.from_numpy(hm1data_padded)
            hm1data = torch.rot90(torch_data,1)
            
            
            #### Headmotion 2 ####
            hm2data, meta = self.loader(hm2_filenum)
            hm2data_padded = np.zeros((256,256,256))
            hm2data_padded[32:224,:,:] = hm2data
            torch_data = torch.from_numpy(hm2data_padded)
            hm2data = torch.rot90(torch_data,1)

            
            
            ################## Loading mask ###################
            maskvolume,meta = self.loader(masklist_filenum)
            maskdata_padded = np.zeros((256,256,256))
            maskdata_padded[32:224,:,:]=maskvolume ### making it 256*256

            ### Rotating it ###
            torch_mask_data = torch.from_numpy(maskdata_padded)
            maskdata = torch.rot90(torch_mask_data,1)

#             mask3d = self.xfms({'image':maskdata})
        
            ###################################################

            #### store volume wise image in a dictionary 
            data_i = {'image':imgdata,'hm1':hm1data,'hm2':hm2data,'mask':maskdata}

            #### transform the data dictionary
            data3d = data_i
            

            if self.cachedir is not None:
                other = {}

                with h5py.File(h5name,'w',libver='latest') as itm:
                    itm.swmr_mode = True
                    for key in data3d:
                        if key in ['image','mask','hm1','hm2']:                             
                            img_npy = data3d[key].numpy()

                            shp = img_npy.shape
                            
                            chunk_size = list(shp[:-1])+[1]
                            ds = itm.create_dataset(key,shp,chunks=tuple(chunk_size),dtype=img_npy.dtype)
                            ds[:]=img_npy[:]
                            ds.flush()
                    else:
                        other[key]=data3d[key]
                torch.save(other,ptname)
                
            
            data = {'image':data3d['image'][:,:,slicenum],'mask':data3d['mask'][:,:,slicenum],'hm1':data3d['hm1'][:,:,slicenum],'hm2':data3d['hm2'][:,:,slicenum]}

            
        if len(data)>0:
            res = {}
            res['image'] = data['image'] 
            res['mask'] = data['mask']
            res['hm1'] = data['hm1']
            res['hm2'] = data['hm2']
            res['filenum'] = filenum
            res['slicenum'] = slicenum
            res['idx'] = index
            return normalize(res['image'][None]) ,normalize(res['hm1'][None]), normalize(res['hm2'][None]),res['mask'][None], {"y":np.array((1))}, vol,slicenum

        else:
            # replace with random
            return self.__getitem__(np.random.randint(len(self.datalist)))
