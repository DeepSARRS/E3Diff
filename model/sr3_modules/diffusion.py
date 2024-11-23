import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        print('======================adopting cosine scheduler========================')
        
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        xT_noise_r=0.1,
        seed = 1,
        opt=None
        
    ):
        super().__init__()
        self.lq_noiselevel_val = schedule_opt["lq_noiselevel"]
        self.opt = opt
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.ddim = schedule_opt['ddim']
        self.xT_noise_r = xT_noise_r
        self.seed = seed
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError() 
        
        

        
    def betas_for_alpha_bar(
        num_diffusion_timesteps,
        max_beta=0.999,
        alpha_transform_type="cosine",
        ):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
        (1-beta) over time from t = [0,1].

        Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
        to that part of the diffusion process.

        Args:
            num_diffusion_timesteps (`int`): the number of betas to produce.
            max_beta (`float`): the maximum beta to use; use values lower than 1 to
                        prevent singularities.
            alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                        Choose from `cosine` or `exp`

        Returns:
            betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
        """
        if alpha_transform_type == "cosine":
            def alpha_bar_fn(t):
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        elif alpha_transform_type == "exp":

            def alpha_bar_fn(t):
                return math.exp(t * -12.0)
        else:
            raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float32)



    def set_new_noise_schedule(self, schedule_opt, device, num_train_timesteps=1000):
        self.ddim = schedule_opt['ddim']
        self.num_train_timesteps = num_train_timesteps
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
                schedule=schedule_opt['schedule'],
                n_timestep=num_train_timesteps,
                linear_start=schedule_opt['linear_start'],
                linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                            to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                            to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                            to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                            to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                            to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                            to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                            to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        
        self.schedule_type = schedule_opt['schedule']
        if self.ddim>0:  # use ddim
            print('================ddim scheduler is adopted===================')
            self.ddim_num_steps = schedule_opt['n_timestep']
            print('==========ddim sampling steps: {}==========='.format(self.ddim_num_steps))
            
            


    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):  # ddpm 采样
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level, t))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance, x_recon




    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def _threshold_sample(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, 0.995, dim=1)
        s = torch.clamp(s, min=1, max=1.0)  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample
    
    def ddim_sample(self, condition_x, img_or_shape, device, seed=1, img_s1=None):
        # self.device = torch.device('cuda:0')
        # self.num_train_timesteps = 2000
        # self.ddim_num_steps = 50
        if self.schedule_type=='linear':
            self.ddim_sampling_eta = 0.8
            simple_var=False
            threshold_x = False   # threshold_x  和 clip_x
        elif self.schedule_type=='cosine':
            self.ddim_sampling_eta = 0.8
            simple_var=False

            threshold_x = False

        # torch.manual_seed(seed)
        batch, total_timesteps, sampling_timesteps, eta= \
                                img_or_shape[0], self.num_train_timesteps, \
                                self.ddim_num_steps, self.ddim_sampling_eta
        # ----------------------------------------------------------------
        
        #----------------conditioned augmentation------------------
        # max_noise_level = 400
        # b = img_s1.shape[0]
        # low_res_noise = torch.randn_like(img_s1).to(img_s1.device)
        # low_res_timesteps = self.lq_noiselevel_val  #
        # lq_noise_level = torch.FloatTensor(
        #             [self.sqrt_alphas_cumprod_prev[low_res_timesteps]]).repeat(b, 1).to(img_s1.device)

        # noisy_img_s1 = self.q_sample(
        #     x_start=img_s1, continuous_sqrt_alpha_cumprod=lq_noise_level.view(-1, 1, 1, 1), noise=low_res_noise)
        noisy_img_s1 = None

        #----------------------------------------------------



        
        if simple_var:
            eta = 1
        ts = torch.linspace(total_timesteps, 0, (sampling_timesteps + 1)).to(device).to(torch.long)

        x = torch.randn(img_or_shape).to(device)
        batch_size = x.shape[0]
        # net = self.denoise_fn
        imgs = [x]
        img_onestep = [condition_x[:,:self.channels,...]]
        if self.opt['stage']!=2:
            tbar = tqdm(range(1, sampling_timesteps + 1),f'seed{seed} DDIM sampling ({self.schedule_type}) with eta {eta} simple_var {simple_var}')
        else:
            tbar = range(1, sampling_timesteps + 1)
        for i in tbar:  
            cur_t = ts[i - 1] - 1
            prev_t = ts[i] - 1
            noise_level = torch.FloatTensor(
                    # [self.sqrt_alphas_cumprod_prev[cur_t+1]]).repeat(batch_size, 1).to(x.device)
                    [self.sqrt_alphas_cumprod_prev[cur_t]]).repeat(batch_size, 1).to(x.device)


            alpha_prod_t = self.alphas_cumprod[cur_t]
            alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else 1
            beta_prod_t = 1 - alpha_prod_t

            # t_tensor = torch.tensor([cur_t] * batch_size,
            #                         dtype=torch.long).to(device).unsqueeze(1)
            # pred noise
            model_output = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level) 

            sigma_2 = eta * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
            noise = torch.randn_like(x)

            # first_term = (alpha_prod_t_prev / alpha_prod_t)**0.5 * x
            # second_term = ((1 - alpha_prod_t_prev - sigma_2)**0.5 -(alpha_prod_t_prev * (1 - alpha_prod_t) / alpha_prod_t)**0.5) * model_output
            # x_start = first_term - (alpha_prod_t_prev * (1 - alpha_prod_t) / alpha_prod_t)**0.5 * model_output
            pred_original_sample = (x - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            
            
            if threshold_x:
                pred_original_sample = self._threshold_sample(pred_original_sample)
            else:
                pred_original_sample = pred_original_sample.clamp(-1, 1)
            
            pred_sample_direction = (1 - alpha_prod_t_prev - sigma_2) ** (0.5) * model_output
            


            if simple_var:
                third_term = (1 - alpha_prod_t / alpha_prod_t_prev)**0.5 * noise  # ddpm使用的方差 
            else:
                third_term = sigma_2**0.5 * noise   # 变成了马尔科夫ddpm
            # x = first_term + second_term + third_term
            x = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction + third_term
            imgs.append(x)
            img_onestep.append(pred_original_sample)

        imgs =  torch.concat(imgs, dim = 0)
        img_onestep =  torch.concat(img_onestep, dim = 0)

        # torch.seed()
        return imgs, img_onestep    
    

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):  # sr3采样
        model_mean, model_log_variance, x_recon = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp(), x_recon

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, seed=1, img_s1=None):
        device = self.betas.device
        # sample_inter = (1 | (self.num_timesteps//20))  
        sample_inter = 1
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            if not self.ddim:
                for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                    img, x_recon = self.p_sample(img, i)
                    if i % sample_inter == 0:
                        ret_img = torch.cat([ret_img, img], dim=0)
            else:
                for i in tqdm(range(0, len(self.ddim_timesteps)), desc='sampling loop time step', total=len(self.ddim_timesteps)):
                    ddim_t = self.ddim_timesteps[i]
                    img = self.ddim_sample(img, ddim_t)
                    if i % sample_inter == 0:
                        ret_img = torch.cat([ret_img, img], dim=0)
                
        else:
            x = x_in
            shape = (x.shape[0], self.channels, x.shape[-2], x.shape[-1])

            # ---------ddpm zT as the inital noise------------------------------------
            if self.xT_noise_r>0:
                # ratio = 0.1
                print('adopting ddpm inversion as initial noise, ratio is {}'.format(self.xT_noise_r))
                img0 = torch.randn(shape, device=device)
                x_start = x_in[:, 0:1, ...]
                continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                                                            np.random.uniform(
                                                                self.sqrt_alphas_cumprod_prev[self.num_timesteps-1],
                                                                self.sqrt_alphas_cumprod_prev[self.num_timesteps],
                                                                size=x_start.shape[0]
                                                            )).to(x_start.device)
                continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(x_start.shape[0], -1)
            
                noise = default(x_start, lambda: torch.randn_like(x_start))
                img = self.q_sample(
                            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)
                
                
                img = self.xT_noise_r*img + (1-self.xT_noise_r)*img0
            #-------------------------------------------------------------------------
            else:
                img = torch.randn(shape, device=device)

            ret_img = x
            img_onestep = x

            if self.opt['stage']!=2:
                if not self.ddim:
                    for i in tqdm(reversed(range(0, self.num_timesteps)), desc='ddpm sampling loop time step', total=self.num_timesteps):
                        img, x_recon = self.p_sample(img, i, condition_x=x)
                        if i % sample_inter == 0:
                            ret_img = torch.cat([ret_img[:,:self.channels,...], img], dim=0)
                        if i % sample_inter==0 or i==self.num_timesteps-1:
                            img_onestep = torch.cat([img_onestep[:,:self.channels,...], x_recon], dim=0)

                else:
                    ret_img, img_onestep = self.ddim_sample(condition_x=x, img_or_shape=shape, device=device, seed=seed, img_s1=img_s1)
            
                
                if continous:
                    return ret_img, img_onestep
                else:
                    return ret_img[-x_in.shape[0]:], img_onestep
            else:
                # timestep = self.num_timesteps-1
                self.ddim_num_steps = self.opt['ddim_steps']
                ret_img, img_onestep = self.ddim_sample(condition_x=x, img_or_shape=shape, device=device, seed=seed, img_s1=img_s1)


                # img, x_recon = self.p_sample(img, timestep, condition_x=x)
                # ret_img = torch.cat([ret_img[:,:self.channels,...], x_recon], dim=0)
                # img_onestep = torch.cat([img_onestep[:,:self.channels,...], x_recon], dim=0)

                if continous:
                    return ret_img, img_onestep
                else:
                    return ret_img[-x_in.shape[0]:], img_onestep

                # for i in tqdm(range(0, len(self.ddim_timesteps)), desc='ddim sampling loop time step', total=len(self.ddim_timesteps)):
                #     ddim_t = self.ddim_timesteps[i]
                #     img = self.ddim_sample(img, ddim_t, condition_x=x)
                #     if i % sample_inter == 0:
                #         ret_img = torch.cat([ret_img[:,:self.channels,...], img], dim=0)
                
                
        #  20, 8, 2hw



    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)


    @torch.no_grad()
    def super_resolution(self, x_in, continous=False, seed=1, img_s1=None):   # 测试

        return self.p_sample_loop(x_in, continous, seed=seed, img_s1=img_s1)






    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        # x_in    {'HR': img_EO[0:1], 'LR': img_s1[0:1], 'condition': img_ppb[0:1], 'SR': img_s1[0:1], 'Index': index, 'filename':filename}
        x_start = x_in['HR']



        [b, c, h, w] = x_start.shape
        if self.opt['stage'] ==2:
            t = 999
            self.ddim_num_steps = self.opt['ddim_steps']
            x = x_in['SR']
            shape = (x.shape[0], self.channels, x.shape[-2], x.shape[-1])
            ret_img, img_onestep = self.ddim_sample(condition_x=x, img_or_shape=shape, device=x.device, seed=self.seed, img_s1=x)
            x_recon = ret_img[-x.shape[0]:]
            

        else:
            t = np.random.randint(1, self.num_timesteps + 1)

            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[t-1],
                    self.sqrt_alphas_cumprod_prev[t],
                    size=b
                )).to(x_start.device)
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

            #-----------pixel loss-------------
            noise = default(noise, lambda: torch.randn_like(x_start))
            x_noisy = self.q_sample(
                x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

 
            ##low_res_timesteps in the paper, they present a new trick where they noise the lowres conditioning image, and at sample time, fix it to a certain level (0.1 or 0.3) - the unets are also made to be conditioned on this noise level
            if not self.conditional:
                x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
            else:
                
                x_recon, condition_feats = self.denoise_fn(
                                                        torch.cat([x_in['SR'], x_noisy], dim=1), 
                                                        continuous_sqrt_alpha_cumprod, 
                                                        # noisy_img_s1, 
                                                        # class_label=lq_continuous_sqrt_alpha_cumprod,
                                                        return_condition=True
                                                        )
        if self.opt['stage']==2:
            l_pix = self.loss_func(x_start, x_recon)    

        else:
            l_pix = self.loss_func(noise, x_recon)    

 
        x_pred = x_recon
        condition_feats=None
 
        
        return l_pix, x_start, x_pred, condition_feats, torch.tensor(t, device=l_pix.device)


    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
