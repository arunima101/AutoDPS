import os
import glob
import numpy as np
import torch
from tqdm.notebook import tqdm
from monai.data import ITKReader,NibabelReader
from monai.transforms import LoadImage, LoadImaged
# from monai.transforms import (
#     Orientationd, AddChanneld, Compose, ToTensord, Spacingd,Resized,ScaleIntensityD,ResizeWithPadOrCropd
#     # ScaleIntensityD, ScaleIntensityRangeD, AdjustContrastD, RandAffined, ToNumpyd,RepeatChannelD
# )
from monai.data import Dataset
import h5py
import threading

## Resized(keys = ('t1','t2','flair','t1ce','seg'),spatial_size = (256, 258,-1),mode = 'trilinear' ,align_corners = True),
def load_data(brats_path,h5_path):
    brats_path = brats_path + '/**/**.nii'
    transforms_1 = None
#     transforms_1 = Compose(
#             [
#              AddChanneld(('t1','t2','flair','t1ce')),
#              Orientationd(('t1','t2','flair','t1ce'),'RAS'),
#              ScaleIntensityD(('t1','t2','flair','t1ce',)),
#              ToTensord(('t1','t2','flair','t1ce')),
#             ]
#         )
    
    dataset = H5CachedDataset(brats_path,transforms_1,h5cachedir=h5_path)
    

    return dataset


mod_list = ['t1', 't2','t1ce','flair'] ##,'seg'
def comb_path(path):
    dict_ = {}
    for itr_ in mod_list:
        uniq_ = path.split('/')[-1].split('_')
        a= "/".join(path.split('/')[0:-1])
        b = '_'.join(uniq_[0:-1])
        c = '/'.join([a,b])
        itr__ = itr_+'.nii'
        d ='_'.join([c,itr__])
        dict_.update({itr_:d})
        
    return dict_

# +
class H5CachedDataset(Dataset):
    def __init__(self, datapath,
                 transforms_1, 
                 nslices_per_image = 155,
                 start_slice = 50,
                 end_slice = 50,
                 h5cachedir=None):

        if h5cachedir is not None:
            if not os.path.exists(h5cachedir):
                os.mkdir(h5cachedir)
            self.cachedir = h5cachedir
        niilist=[]
        
        for x in glob.glob(datapath):
            niilist.append(comb_path(x))

        self.datalist = niilist
#         print(self.datalist)
#         self.masklist = masklist
#         print(f'length of datalist {len(niilist)}')
#         print(f'length of masklist {len(masklist)}')
        
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
        
        filenum = index // (self.nslices - self.start_slice)

        slicenum = index % (self.nslices - self.start_slice)

        slicenum += self.start_slice
        
        #### Extract the datafile location & mask file location based on filenum
        datalist_filenum = self.datalist[filenum]

        loc_data_t1 = datalist_filenum['t1']
        loc_data_t2 = datalist_filenum['t2']
        loc_data_flair = datalist_filenum['flair']
        loc_data_t1ce = datalist_filenum['t1ce']
#         loc_data_seg = datalist_filenum['seg']


        ##### if h5 exists for the current volume fill data dictionary with current slice number
        if self.cachedir is not None:
            h5name = self.cachedir+'/%d.h5' % filenum
            vol = int(h5name.split("/")[-1].split(".")[0])
            ptname = self.cachedir+'/%d.pt' % filenum
            
            if os.path.exists(h5name):
                
                with h5py.File(h5name,'r',libver='latest', swmr=True) as itm:
                    for key in itm.keys(): 
#                         volume = torch.rot90(itm[key],1)
                        data[key]=torch.from_numpy(itm[key][:,:,:,slicenum])
#                 data['image_meta_dict']={'affine':np.eye(3)} # FIXME: loses spacing info - can read from pt file


        ##### if data dictionary is empty
        if len(data)==0:
            #### Read image & meta data

            imgdata_t1, meta_t1 = self.loader(loc_data_t1)
            imgdata_t1_padded = torch.zeros((256,256,155))
            imgdata_t1_padded[8:-8,8:-8,:] = imgdata_t1

            
            
            imgdata_t2, meta_t2 = self.loader(loc_data_t2)
            imgdata_t2_padded = torch.zeros((256,256,155))
            imgdata_t2_padded[8:-8,8:-8,:] = imgdata_t2
            
            imgdata_t1ce, meta_t1ce = self.loader(loc_data_t1ce)
            imgdata_t1ce_padded = torch.zeros((256,256,155))
            imgdata_t1ce_padded[8:-8,8:-8,:] = imgdata_t1ce
            
            imgdata_flair, meta_flair = self.loader(loc_data_flair)
            imgdata_flair_padded = torch.zeros((256,256,155))
            imgdata_flair_padded[8:-8,8:-8,:] = imgdata_flair
            
#             imgdata_seg, meta_seg = self.loader(loc_data_seg)
#             imgdata_seg_padded = torch.zeros((256,256,155))
#             imgdata_seg_padded[8:-8,8:-8,:] = imgdata_seg
            
            
            
            #### store volume wise image & mask data,metadata in a dictionary 
            data_i = {'t1':imgdata_t1_padded,\
                      't2':imgdata_t2_padded,\
                      't1ce':imgdata_t1ce_padded,\
                      'flair':imgdata_flair_padded,\
#                       'seg':imgdata_seg_padded,\
                      
                      't1_meta_dict':meta_t1, \
                      
                      't2_meta_dict':meta_t2, \
                      't1ce_meta_dict':meta_t1ce,\
                      'flair_meta_dict':meta_flair
#                       'seg_meta_dict':meta_seg
                     }

            #### transform the data dictionary
            data3d = self.xfms(data_i)

            if self.cachedir is not None:
                other = {}

                with h5py.File(h5name,'w',libver='latest') as itm:
                    itm.swmr_mode = True
                    for key in data3d:
                        if key in ['t1','t2','flair','t1ce']:    ##,'seg'                         
                            img_npy = data3d[key].numpy()
                            shp = img_npy.shape
                            
                            chunk_size = list(shp[:-1])+[1]

                            ds = itm.create_dataset(key,shp,chunks=tuple(chunk_size),dtype=img_npy.dtype)
                            ds[:]=img_npy[:]

                            ds.flush()
                    else:
                        other[key]=data3d[key]
                torch.save(other,ptname)


            #### fill the data dictionary

            data = {'t1':data3d['t1'][:,:,:,slicenum],\
                    't2':data3d['t2'][:,:,:,slicenum],\
                    't1ce':data3d['t1ce'][:,:,:,slicenum],\
                    'flair':data3d['flair'][:,:,:,slicenum]}
#                     'seg':data3d['seg'][:,:,:,slicenum],}

            
        if len(data)>0:

            res = data

            res['t1']=res['t1'].float()
            res['t2']=res['t2'].float()
            res['t1ce']=res['t1ce'].float()
            res['flair']=res['flair'].float()
#             res['seg']=res['seg'].float()
            
            res['filenum'] = filenum
            res['slicenum'] = slicenum
            res['idx'] = index
            mask = res['t2']>0
            return res['t2'],mask,vol,slicenum

        else:
            # replace with random
            return self.__getitem__(np.random.randint(len(self.datalist)))
