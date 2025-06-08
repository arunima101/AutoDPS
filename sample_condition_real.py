from functools import partial
import os
import argparse
import yaml
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from scipy.signal import wiener
from skimage import color, data, restoration
import math 

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler

##### For autofocusing ==================================
from guided_diffusion.demotion_real import autofocusing
from torch.fft import fftshift, ifftshift, fftn, ifftn

Ft = lambda x : torch.fft.fftshift(torch.fft.fft2(x))
IFt = lambda x : torch.fft.ifftn(torch.fft.ifftshift(x))
###=======================================================


# ############# For undersampling ##########

from guided_diffusion.mask_set import create_mask

# ########################################

# from data.dataloader import get_dataset, get_dataloader

import sys
sys.path.insert(0,'/storage/arunima/Arunima/diffusion_codes/DPS_using_ddim/AutoDPS/data')
from data import * 

# from data.load_brats import load_data   ## added

from load_mr_art import load_data

from util.img_utils import clear_color, mask_generator
from util.logger import get_logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def normalize(image):
    return (image - image.min())/(image.max()-image.min())

def psnr(original, contrast):
    mse = np.mean((original - contrast) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR

# +
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    
    avg_psnr = []
    avg_psnr_fil=[]
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser,measure_config['operator']['name'],\
                                          **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    if diffusion_config['sampler']=='ddim':
        sample_fn = partial(sampler.ddim_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
    else:
        sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    #  Working directory
    save_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    

    # Prepare dataloader

    data_path = task_config['data']['data_path']
    h5_path = task_config['data']['h5_path']
    mode = task_config['data']['mode']
    motion_level = task_config['data']['motion_level']
    
    dataset = load_data(data_path,h5_path,mode = mode)

    loader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle= False ,num_workers = 0)
    

    # In case of undersampling, we need to generate a mask 
    if measure_config['operator']['name'] == 'undersampling' or measure_config['operator']['name'] =='motion+undersampling':
        
        mask_type = measure_config['operator']['masktype']
        acc = measure_config['operator']['accfactor']
        a = torch.ones((1,1,256,256))
        m=create_mask(mask_type,a.shape,acc).to(device)

    # Do Inference
    
    for i,img in enumerate(loader):
        
        
        logger.info(f"Inference for image {i}")
        mask = img[3].to(device)
        ref_img = img[0].to(device)*mask
        if motion_level ==1:
            y = img[1].to(device)*mask
        else:
            y = img[2].to(device)*mask
        
        vol = int(img[-2][0])
        slice = int(img[-1][0])
        
        os.makedirs(args.save_dir, exist_ok=True)
        
        out_path = save_path + f'/image_{vol}_{slice}'

        for img_dir in ['input', 'recon', 'progress',\
                        'degradation','clean_image',\
                        'norm_grad','x_0_hat','filtered','mask']:
            os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)


        fname = str(i).zfill(5) + '.npz'
        ref_img = ref_img.to(device)

        #  In case of undersampling,
        if measure_config['operator'] ['name'] == 'undersampling':

#             Forward measurement model (Ax)
            y = operator.forward(ref_img, mask=m)
            y_n = noiser(y)
            
            
        elif measure_config['operator']['name'] == 'motion+undersampling':
            
            
            #### Forward model (Ax) A = MOTION-->UNDERSAMPLE
            clean = ref_img
            
            corrupted_ksp,shift_vector,rot_vectorgt = operator.forward_motion(ref_img)
            y1 = abs(IFt(corrupted_ksp))
            

            y = operator.forward_undersample(y1[None,:,:,:], mask=m)
            corrupted_ksp_ = Ft(y)
            
            if noiser.sigma:
#                 print("noise added!")
                y = noiser(y)
                corrupted_ksp_ = Ft(y)[:,:,:,:]
            
            
            ### Applying Autofocusing ####
            
            filtered_image,shift_vec,rotaf = autofocusing(corrupted_ksp_)
            np.savez(os.path.join(out_path, 'filtered', 'image'), clear_color(filtered_image))
            plt.imsave(os.path.join(out_path, 'filtered', 'image.png') , clear_color(filtered_image), cmap='gray')
            
            
            ksp_sub_optimal = Ft(filtered_image)
            ycr = corrupted_ksp_.abs() ### corrupted ksp
            yf = ksp_sub_optimal.abs() ### AF filtered ksp
            yfinv = torch.reciprocal(yf)
            
            anglecr = (1j*corrupted_ksp_.angle()).exp()
            anglef = (-1j*ksp_sub_optimal.angle()).exp()

            new_lhs = yfinv*anglef*ycr*anglecr
            new_phase_shift = new_lhs.angle() ##### estimated A for motion
            
            ##############################

            

            ############ Using GT ####################
            x_shifts = shift_vector[0]
            y_shifts = shift_vector[1]
            x_shape, y_shape = corrupted_ksp.shape[-2:]
            # Translation
            phase_shiftgt = -2 * math.pi * (
                x_shifts * torch.linspace(0, 1, x_shape)[None, :, None] + 
                y_shifts * torch.linspace(0, 1, y_shape)[None, None, :])[0]
            ###########################################

        else: 
            
            # Forward measurement model (Ax + n)
            clean = ref_img.to(device)
            corrupted_ksp = Ft(y)[0,0].to(device)
            


            filtered_image,shift_vec,rotaf = autofocusing(corrupted_ksp) ### send (X,Y) data
            np.savez(os.path.join(out_path, 'filtered', 'image'), clear_color(filtered_image))
            plt.imsave(os.path.join(out_path, 'filtered', 'image.png') , clear_color(filtered_image), cmap='gray')
            
            ksp_sub_optimal = Ft(filtered_image)
            ycr = corrupted_ksp.abs() ### corrupted ksp
            yf = ksp_sub_optimal.abs() ### AF filtered ksp
            yfinv = torch.reciprocal(yf)
            
            anglecr = (1j*corrupted_ksp.angle()).exp()
            anglef = (-1j*ksp_sub_optimal.angle()).exp()

            new_lhs = yfinv*anglef*ycr*anglecr
            new_phase_shift = new_lhs.angle() ##### estimated A


#             y_n = noiser(y)

        #### Sampling ####
    
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        
        if measure_config['operator']['name'] =='motion_artifact_real':
            sample = sample_fn(x_start=x_start,measurement=y,shift=new_phase_shift,\
                               rot=rotaf,noise =0,mask=None,record=True,save_root=out_path)
            
        if measure_config['operator']['name'] == 'undersampling':
            sample = sample_fn(x_start=x_start,measurement=y,shift=None,rot=None,noise = noiser.sigma,mask=m,record=True,save_root=out_path)
            
        if measure_config['operator'] ['name'] == 'motion+undersampling':
            sample = sample_fn(x_start=x_start,measurement=y,shift=new_phase_shift,rot=rotaf,\
                               mask=m,noise = 0,record=True,save_root=out_path)
        
        np.savez(os.path.join(out_path, 'degradation','image'), clear_color(y))
        plt.imsave(os.path.join(out_path , 'degradation','image.png') , clear_color(y), cmap='gray')
        
        np.savez(os.path.join(out_path, 'mask', 'image'), clear_color(mask))
        plt.imsave(os.path.join(out_path , 'mask','image.png') , clear_color(mask), cmap='gray')
        
        np.savez(os.path.join(out_path, 'recon', 'image'), clear_color(sample))
        plt.imsave(os.path.join(out_path , 'recon','image.png'), clear_color(sample), cmap='gray')
        
        np.savez(os.path.join(out_path, 'clean_image', 'image'), clear_color(clean))
        plt.imsave(os.path.join(out_path,'clean_image','image.png'), clear_color(clean), cmap='gray')
        
        del out_path
        


if __name__ == '__main__':
    main()
