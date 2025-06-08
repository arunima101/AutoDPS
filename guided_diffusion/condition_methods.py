from abc import ABC, abstractmethod
import torch
from skimage import color, data, restoration
import numpy as np
import torch 
__CONDITIONING_METHOD__ = {}



def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, deg,**kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser,deg = deg, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser,deg, **kwargs):
        self.operator = operator
        self.noiser = noiser
        self.degradation = deg
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement,shift_vec,rot,mask,noise,**kwargs):
        device = x_prev.device
#         print(self.degradation)
        if self.noiser.__name__ == 'gaussian':
            
            if self.degradation =='motion_artifact_real' or self.degradation =='motion_artifact':
                
                ax = self.operator.corrupt(x_0_hat,shift_vec,rot,**kwargs)
#                 print(noise)
#                 if noise:
#                     ax = self.noiser(ax)            
                difference = measurement - ax
                norm = torch.linalg.norm(difference)       
                norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
                
            elif self.degradation =='undersampling':
                
                difference = measurement - self.operator.forward(x_0_hat,mask,**kwargs)
                norm = torch.linalg.norm(difference)       
                norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
                
            elif self.degradation == 'motion+undersampling':                
                
                motion_cr_x_0 = self.operator.forward_corrupt(x_0_hat,shift_vec,rot,**kwargs)
                un_x_0_hat = self.operator.forward_undersample(motion_cr_x_0,mask,**kwargs)
                if noise:
                    noisy = self.noiser(un_x_0_hat)
                else:
                    noisy = un_x_0_hat
                
                difference = measurement - noisy
                norm = torch.linalg.norm(difference)       
                norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
                            
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale

        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser,deg, **kwargs):
        super().__init__(operator, noiser,deg)
        self.scale = kwargs.get('scale', 1.0)
        

    def conditioning(self, x_prev, x_t, x_0_hat, measurement,shift_vec,rot,mask,noise,**kwargs):
        norm_grad, norm= self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement,\
                                             shift_vec=shift_vec,rot = rot,mask=mask,noise=noise,**kwargs)
        x_t -= norm_grad * self.scale
        
        return x_t, norm, norm_grad
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm
