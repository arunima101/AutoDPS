import os
import glob
import numpy as np
import torch
from tqdm.notebook import tqdm
from monai.data import ITKReader,NibabelReader
from monai.transforms import LoadImage, LoadImaged
from monai.transforms import (
    Orientationd, AddChanneld, Compose, ToTensord, Spacingd,Resized,ScaleIntensityD,ResizeWithPadOrCropd
    # ScaleIntensityD, ScaleIntensityRangeD, AdjustContrastD, RandAffined, ToNumpyd,RepeatChannelD
)
from monai.data import Dataset
import h5py
import threading


def load_data(brats_path,h5_path):
    brats_path = brats_path + '/**/**_t1.nii'
    transforms_1 = Compose(
            [
             AddChanneld(('image_t1','image_t2')),
             Orientationd(('image_t1','image_t2'),'RAS'),
             ScaleIntensityD(('image_t1','image_t2',)),
             ToTensord(('image_t1','image_t2')),
            ]
        )
    
    dataset = H5CachedDataset(brats_path,transforms_1,h5cachedir=h5_path)
    

    return dataset
    

mod_list = ['t1', 't2']
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

class H5CachedDataset(Dataset):
    def __init__(self, datapath,
                 transforms_1, 
                 nslices_per_image = 155,
                 start_slice = 50,
                 end_slice = 50,
                 h5cachedir=None):
       #         self.lock = threading.Lock()
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
#         self.xfms2 = transforms_online

            
        #### 3d image loader from monai
        
        self.loader = LoadImage()
        self.loader.register(NibabelReader())  
        #### start_slice & end_slice---> slices to be truncated in each volume vol[:,:,start_slice:-end_slice]
        
        self.start_slice = start_slice
        self.end_slice = end_slice
#         print(self.end_slice)
#         print(nslices_per_image)
        self.nslices = nslices_per_image - self.end_slice
#         print(self.nslices)
        
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
#         print("index",index)
#        
        filenum = index // (self.nslices - self.start_slice)
#         print(index)

        slicenum = index % (self.nslices - self.start_slice)

        slicenum += self.start_slice
        
        #### Extract the datafile location & mask file location based on filenum
        datalist_filenum = self.datalist[filenum]
#         'flair', 't1', 't2', 't1ce','seg'
#         loc_data_flair = datalist_filenum['flair']###path to image
        loc_data_t1 = datalist_filenum['t1']
        loc_data_t2 = datalist_filenum['t2']
#         print(loc_data_t1)
#         loc_data_t1ce = datalist_filenum['t1ce']
#         loc_mask = datalist_filenum['seg'] ### path to label
        
        


        ##### if h5 exists for the current volume fill data dictionary with current slice number
        if self.cachedir is not None:
            h5name = self.cachedir+'/%d.h5' % filenum
#             print(h5name)
            ptname = self.cachedir+'/%d.pt' % filenum

            if os.path.exists(h5name):
                
                with h5py.File(h5name,'r',libver='latest', swmr=True) as itm:
                    for key in itm.keys():                       
                        data[key]=torch.from_numpy(itm[key][:,:,:,slicenum])
                data['image_meta_dict']={'affine':np.eye(3)} # FIXME: loses spacing info - can read from pt file


        ##### if data dictionary is empty
        if len(data)==0:
            #### Read image & mask data, meta data
#             imgdata_flair, meta_flair = self.loader(loc_data_flair) 
            imgdata_t1, meta_t1 = self.loader(loc_data_t1)
            imgdata_t1_padded = torch.zeros((256,256,155))
            imgdata_t1_padded[8:-8,8:-8,:] = imgdata_t1
#             print("t1 shape",imgdata_t1_padded)
            
            
            imgdata_t2, meta_t2 = self.loader(loc_data_t2)
            imgdata_t2_padded = torch.zeros((256,256,155))
            imgdata_t2_padded[8:-8,8:-8,:] = imgdata_t2
#             print("t2 shape",imgdata_t2_padded.shape)
#             print("t1 shape",imgdata_t1.shape)
#             imgdata_t1ce, meta_t1ce = self.loader(loc_data_t1ce) 
            
#             mask_data, mask_meta = self.loader(loc_mask)
            
            #### store volume wise image & mask data,metadata in a dictionary 
            data_i = {'image_t1':imgdata_t1_padded,\
                      'image_t2':imgdata_t2_padded,\
                      
                      't1_meta_dict':meta_t1, \
                      
                      't2_meta_dict':meta_t2, \
                      
                     }
#             print(imgdata.shape) (240*240*155)
            #### transform the data dictionary
            data3d = self.xfms(data_i)
#             print(data3d['image'].shape)
            #### Create h5 file for the volume by chunking into the slice shape for data & mask 
            #### Create a .pt file for meta data
            if self.cachedir is not None:
                other = {}

                with h5py.File(h5name,'w',libver='latest') as itm:
                    itm.swmr_mode = True
                    for key in data3d:
                        if key in ['image_t1','image_t2']:                             
                            img_npy = data3d[key].numpy()
#                             if key == 'label':
#                                 img_npy = (img_npy>0).astype('uint8')
                            shp = img_npy.shape
                            
                            chunk_size = list(shp[:-1])+[1]
#                             print(chunk_size)
                            ds = itm.create_dataset(key,shp,chunks=tuple(chunk_size),dtype=img_npy.dtype)
                            ds[:]=img_npy[:]
#                             print('ds shape',ds.shape)
                            ds.flush()
                    else:
                        other[key]=data3d[key]
                torch.save(other,ptname)


            #### fill the data dictionary
#             print(f'slicenum {slicenum}')
#             print(f'filenum {filenum}')
#             ['image_flair','image_t1','image_t2','image_t1ce','label']
            data = {'image_t1':data3d['image_t1'][:,:,:,slicenum],\
                'image_t2':data3d['image_t2'][:,:,:,slicenum],\
                
                'image_meta_dict':{
                    'affine':np.eye(3)
                }
            }

            
        if len(data)>0:
#             print("**")
#             res = self.xfms2(data)
            res = data
# #             print(data['image_t1'])
            res['image_t1']=res['image_t1'].float()
            res['image_t2']=res['image_t2'].float()
            res['filenum'] = filenum
            res['slicenum'] = slicenum
            res['idx'] = index
            return res

        else:
            # replace with random
            return self.__getitem__(np.random.randint(len(self.datalist)))
