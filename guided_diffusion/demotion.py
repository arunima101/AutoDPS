import torch
from torch.fft import fftshift, ifftshift, fftn, ifftn

Ft = lambda x : torch.fft.fftshift(torch.fft.fft2(x))
IFt = lambda x : torch.fft.ifftn(torch.fft.ifftshift(x))
import numpy as np
import math

def entropy_criterion(image):
    img = image[0]
    bmax = torch.sqrt(torch.sum(img**2))
    x,y = img.shape
    lst = 0
    for i in range(x-1):
        for j in range(y-1): 
            val = (img[i,j]/bmax)*torch.log(img[i,j]/bmax)            
            lst = lst+val
            
    ans = -1*lst
    return ans

# def entropy_criterion(image):
#     img = image[0].to(image.device)  # Move image to GPU (assuming 'device' is set to the appropriate GPU)
#     bmax = torch.sqrt(torch.sum(img**2))
#     img_normalized = img / bmax
#     img_log = torch.log(img_normalized)
#     img_log[img_normalized == 0] = 0  # Handling log(0) cases
#     entropy = -torch.sum(img_normalized * img_log)
#     return entropy

# +
def rotate_image(img,theta):
#     print(img.shape)
    lenx, leny = img.shape[-2:]
    img_rot = torch.zeros_like(img)

#     theta = torch.tensor(0.01)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    # Create grid coordinates
    x = torch.linspace(-lenx//2, lenx//2, lenx).to(img.device)
    y = torch.linspace(-leny//2, leny//2, leny).to(img.device)
    grid_x, grid_y = torch.meshgrid(x, y)

    # Apply rotation transformation
    xt = grid_x * cos_theta - grid_y * sin_theta + lenx // 2
    yt = grid_y * cos_theta + grid_x * sin_theta + leny // 2


    # Calculate indices for interpolation
    xf = torch.floor(xt).long().clamp(0, lenx - 2)
    yf = torch.floor(yt).long().clamp(0, leny - 2)
    xc = xf + 1
    yc = yf + 1
    a = xt - xf.float()
    b = yt - yf.float()

    # Apply bilinear interpolation
    
    img_rot[:, :, :] = (
        (1 - a) * (1 - b) * img[:, xf, yf] +
        (1 - a) * b * img[:, xf, yc] +
        a * (1 - b) * img[:, xc, yf] +
        a * b * img[:, xc, yc]
    )

    return img_rot


# -

def autofocusing(ks):
    beta1, beta2 = 0.89, 0.8999
    ps = ks.shape[-1]
    ps_cf = int((ps // 2) * 0.08)
    zero_middle = torch.ones((ps)).cuda()
    zero_middle[ps // 2 - ps_cf : ps // 2 + ps_cf] = 0.
    img = IFt(ks).abs()

    # Translation Params
    x_shifts = torch.zeros(ps)
    y_shifts = torch.zeros(ps)
    x_shifts = torch.nn.Parameter(data=x_shifts.cuda(), requires_grad=True)
    y_shifts = torch.nn.Parameter(data=y_shifts.cuda(), requires_grad=True)
    x_moment1 = torch.nn.Parameter(data=torch.zeros_like(x_shifts), requires_grad=True)
    x_moment2 = torch.nn.Parameter(data=torch.zeros_like(x_shifts), requires_grad=True)
    y_moment1 = torch.nn.Parameter(data=torch.zeros_like(y_shifts), requires_grad=True)
    y_moment2 = torch.nn.Parameter(data=torch.zeros_like(y_shifts), requires_grad=True)
    # Rotation Params
    rot_vector = torch.tensor(0.0).cuda()
    rot_vector = torch.nn.Parameter(data=rot_vector.cuda(), requires_grad=True)
    rot_moment1 = torch.nn.Parameter(data=torch.zeros_like(rot_vector), requires_grad=True)
    rot_moment2 = torch.nn.Parameter(data=torch.zeros_like(rot_vector), requires_grad=True)
    
    for _ in range(80):
#         rot_vector = rot_vector * zero_middle
        x_shifts = x_shifts * zero_middle
        y_shifts = y_shifts * zero_middle
        # Translation
        phase_shift = -2 * math.pi * (
            x_shifts * torch.linspace(0, ps, ps)[None, :, None].cuda() + 
            y_shifts * torch.linspace(0, ps, ps)[None, None, :].cuda())[0]
        yp_ks = ks.abs().cuda() * (1j * (ks.angle().cuda() + phase_shift)).exp()
        img = IFt(yp_ks).abs()
        
        # Rotation

        yp_img = rotate_image(img,rot_vector).cuda()
    

        loss_net = (yp_img[None] * 1e4).mean()
#         loss_net = entropy_criterion(yp_img)
        x_grad, y_grad,rot_grad = torch.autograd.grad(loss_net, [x_shifts, y_shifts,rot_vector],
                                                       create_graph=False)
        x_grad, y_grad , rot_grad = x_grad * 1e-4, y_grad * 1e-4 , rot_grad * 1e-4
        x_moment1 = beta1 * x_moment1 + (1. - beta1) * x_grad
        x_moment2 = beta2 * x_moment2 + (1. - beta2) * x_grad * x_grad + 1e-24
        y_moment1 = beta1 * y_moment1 + (1. - beta1) * y_grad
        y_moment2 = beta2 * y_moment2 + (1. - beta2) * y_grad * y_grad + 1e-24
        x_shifts = x_shifts - 3e-4 * x_moment1 * x_moment2.rsqrt()
        y_shifts = y_shifts - 3e-4 * y_moment1 * y_moment2.rsqrt()
        rot_moment1 = beta1 * rot_moment1 + (1. - beta1) * rot_grad
        rot_moment2 = beta2 * rot_moment2 + (1. - beta2) * rot_grad * rot_grad + 1e-24
        rot_vector = rot_vector - 3e-4  * rot_moment1 * rot_moment2.rsqrt()
 

    x_shifts = x_shifts * zero_middle
    y_shifts = y_shifts * zero_middle
    
    shift_vec= torch.empty((2, 256))
    shift_vec[0]=x_shifts
    shift_vec[1]=y_shifts
    
    # Translation
    phase_shift = -2 * math.pi * (
        x_shifts * torch.linspace(0, ps, ps)[None, :, None].cuda() + 
        y_shifts * torch.linspace(0, ps, ps)[None, None, :].cuda())[0]
    yp_ks = ks.abs().cuda() * (1j * (ks.angle().cuda() + phase_shift)).exp()
    img = IFt(yp_ks).abs() ### 
    # Rotation
    yp_img = rotate_image(img,rot_vector).cuda()
    
        
    return yp_img,phase_shift,rot_vector
