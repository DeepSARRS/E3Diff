import logging
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')
import model.lr_scheduler as lr_scheduler
from model.sr3_modules.perceptual import PerceptualLoss
from model.sr3_modules.focal_frequency_loss import FocalFrequencyLoss
import lpips
import vision_aided_loss



class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()

        # 将freq初始化为0
        # for k, v in self.netG.named_parameters():
        #     # print(k)
        #     if 'freq' in k:
        #         v.data.zero_()
        #         # print(v)
        #         print(f'setting {k} to zeros')



        print('--------------------setting Perceptual loss-------------------------')

        self.lpips_loss = lpips.LPIPS(net='vgg', spatial=True).to(self.device)  # RGB, normalized to [-1,1]
        for k, v in self.lpips_loss.named_parameters():
            v.requires_grad = False
        

        
        self.net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type='multilevel_sigmoid_s', device="cuda")
        self.net_disc = self.net_disc.cuda()
        self.net_disc.requires_grad_(True)
        self.net_disc.cv_ensemble.requires_grad_(False)
        self.net_disc.train()

        self.optimizer_disc = torch.optim.Adam(self.net_disc.parameters(), lr=opt['train']["optimizer"]["lr"])
        
        print('--------------------setting FocalFrequency loss-------------------------')
        self.freq_loss = FocalFrequencyLoss(loss_weight=opt['loss_w']['fft_w'], alpha=1.0)
        print('--------------------------------------------------------------------')


        self.set_new_noise_schedule(
            opt['model']['beta_schedule'][self.opt['phase']], schedule_phase=self.opt['phase'], 
            num_train_timesteps = opt['model']['beta_schedule']['train']['n_timestep'])
        
        self.load_network()
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()

        self.print_network()
        if self.opt['phase'] == 'train':
            self.setup_schedulers()


    def setup_schedulers(self, lastepoch=-1):
        """Set up scheduler."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        self.scheduler_type = scheduler_type
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            self.scheduler=lr_scheduler.MultiStepRestartLR(self.optG,
                                                    **train_opt['scheduler'], last_epoch=lastepoch)
        elif scheduler_type == 'CosineAnnealingRestartLR':
            self.scheduler=lr_scheduler.CosineAnnealingRestartLR(
                        self.optG, **train_opt['scheduler'], last_epoch=lastepoch)
        elif scheduler_type == 'CosineAnnealingWarmupRestarts':
            self.scheduler=lr_scheduler.CosineAnnealingWarmupRestarts(
                        self.optG, **train_opt['scheduler'], last_epoch=lastepoch)
        elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
            self.scheduler=lr_scheduler.CosineAnnealingRestartCyclicLR(
                        self.optG, **train_opt['scheduler'], last_epoch=lastepoch)
        elif scheduler_type == 'TrueCosineAnnealingLR':
            print('..', 'cosineannealingLR')
            self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optG, **train_opt['scheduler'], last_epoch=lastepoch)
        elif scheduler_type == 'CosineAnnealingLRWithRestart':
            print('..', 'CosineAnnealingLR_With_Restart')
            self.scheduler=lr_scheduler.CosineAnnealingLRWithRestart(self.optG, **train_opt['scheduler'], last_epoch=lastepoch)
        elif scheduler_type == 'LinearLR':
            self.scheduler=lr_scheduler.LinearLR(
                        self.optG, train_opt['n_iter'], last_epoch=lastepoch)
        elif scheduler_type == 'VibrateLR':
            self.scheduler=lr_scheduler.VibrateLR(
                        self.optG, train_opt['n_iter'], last_epoch=lastepoch)
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')
        print('Setting up Scheduler finished')

    
    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.
        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warm-up iter numbers. -1 for no warm-up.
                Default： -1.
        """
        if current_iter > 1:
            self.scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optG.param_groups]


    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix, x_start, x_pred, condition_feats, tt = self.netG(self.data)
        
        
        
        self.SR_onestep = x_pred

        t_loss_weight = (1 - tt/self.opt['model']['beta_schedule']['train']['n_timestep']) ** 2

        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum()/int(b*c*h*w)

        if self.opt["loss_w"]["lpips_w"]>0:
            lpips_errs = self.lpips_loss(x_pred, x_start)  # [bs,1,h,w]
            lpips_loss = lpips_errs.sum()/int(b*h*w)* self.opt["loss_w"]['lpips_w']
        else:
            lpips_loss = torch.zeros_like(l_pix)

        if self.opt["loss_w"]["fft_w"]>0:
            freq_loss = self.freq_loss(x_pred, x_start) #*  t_loss_weight.sum()
        else:
            freq_loss = torch.zeros_like(l_pix)
            
        loss = l_pix + lpips_loss + freq_loss # + condition_loss

        if not torch.isnan(l_pix).any():
            loss.backward()
            self.optG.step()

            # set log
            self.log_dict['l_pix'] = l_pix.item()
            self.log_dict['l_per'] = lpips_loss.sum().item()
            self.log_dict['l_freq'] = freq_loss.sum().item()

        else:
            print('loss is nan')
        
        if self.opt["stage"]==2:
            """
            Generator loss: fool the discriminator
            """
            self.optG.zero_grad()
            l_pix, x_start, x_tgt_pred, condition_feats, tt = self.netG(self.data)
            # x_tgt_pred = net_pix2pix(x_src, prompt_tokens=batch["input_ids"], deterministic=True)
            lossG = self.net_disc(x_tgt_pred, for_G=True).mean() * self.opt["loss_w"]['lambda_gan']
            lossG.backward()
            self.optG.step()
            self.optG.zero_grad()
            """
            Discriminator loss: fake image vs real image
            """
            # real image
            lossD_real = self.net_disc(x_start.detach(), for_real=True).mean() * self.opt["loss_w"]['lambda_gan']
            lossD_real.mean().backward()
            self.optimizer_disc.step()
            self.optimizer_disc.zero_grad()
            # fake image
            lossD_fake = self.net_disc(x_tgt_pred.detach(), for_real=False).mean() * self.opt["loss_w"]['lambda_gan']
            lossD_fake.mean().backward()
            self.optimizer_disc.step()
            self.optimizer_disc.zero_grad()
            lossD = lossD_real + lossD_fake

            self.log_dict['l_G'] = lossG.item()
            self.log_dict['l_D'] = lossD.item()




    def test(self, continous=False, seed=1):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR, self.SR_onestep = self.netG.module.super_resolution(
                    self.data['SR'], continous, seed, self.data['LR'])
            else:
                self.SR, self.SR_onestep = self.netG.super_resolution(
                    self.data['SR'], continous, seed, self.data['LR'])
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train', num_train_timesteps=1000):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device, num_train_timesteps)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device, num_train_timesteps)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            out_dict['SR_onestep'] = self.SR_onestep.detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            net_params = torch.load(gen_path)



            network.load_state_dict(net_params, strict=False)
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                # self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
