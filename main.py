'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-12-24 12:34:50
LastEditors: Please set LastEditors
LastEditTime: 2024-11-23 14:19:08
FilePath: /QJ/E3Diff/train_call.py
'''
import time
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SAR2EO_256_s1.json',   
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', default='false')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('--xT_noise_r', type=float, help='the ratio of ddpm inversion as initial noise', default=0.)
    parser.add_argument('--seed', type=float, help='the ratio of ddpm inversion as initial noise', default=1)

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    opt['xT_noise_r'] = args.xT_noise_r
    opt['seed'] = int(args.seed)
    
    torch.cuda.manual_seed(opt['seed'])
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # opt['enable_wandb'] = False
    # Initialize WandbLogger
    if opt['enable_wandb'] and opt['phase']=='train':
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None
        

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    cal_metrics = Metrics.cal_metrics(diffusion.device)
    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))
        if opt['phase'] == 'train':
            logger.info('Setting scheduler as: {}, last iter: {}.'.format(
                    diffusion.scheduler_type, current_step))
            diffusion.scheduler.last_epoch = current_step


    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], 
        schedule_phase=opt['phase'], 
        num_train_timesteps=opt['model']['beta_schedule']['train']['n_timestep'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for data_it, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                diffusion.update_learning_rate(current_step)
                
                # validation
                if data_it ==0:
                    train_path = os.path.join(opt['path']['results'], 'train_img')
                    os.makedirs(train_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                                        opt['model']['beta_schedule']['val'], schedule_phase='val',
                                        num_train_timesteps=opt['model']['beta_schedule']['train']['n_timestep'])
                    
                    diffusion.feed_data(train_data)
                    diffusion.test(continous=False)
                    visuals = diffusion.get_current_visuals()
                    sample_img = Metrics.tensor2img(visuals['SR'])  # uint8     samples
                    tgt_img = Metrics.tensor2img(visuals['HR'])  # uint8     groundtruth
                    lq_img = Metrics.tensor2img(visuals['INF'][:,:1,...])
                    SR_onestep = Metrics.tensor2img(visuals['SR_onestep'])

                    if current_step%100000==0:
                        Metrics.save_img(
                            SR_onestep, '{}/{}_{}_onestep.png'.format(train_path, current_epoch, data_it))
                    
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train',
                        num_train_timesteps=opt['model']['beta_schedule']['train']['n_timestep'])
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.4e}> '.format(
                        current_epoch, current_step,  diffusion.get_current_learning_rate()[0])
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)


                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    avg_lpips = 0.0

                    idx = 0
                    result_path = opt['path']['results']
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val',
                        num_train_timesteps=opt['model']['beta_schedule']['train']['n_timestep'])
                    for vi,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        sample_img = Metrics.tensor2img(visuals['SR'])  # uint8    generated samples
                        tgt_img = Metrics.tensor2img(visuals['HR'])  # uint8     target image
                        src_img = Metrics.tensor2img(visuals['LR'])  # uint8    source image


                        if vi % 5 ==0:
                            tb_logger.add_image(
                                'Iter_{}'.format(current_step),
                                np.transpose(np.concatenate(
                                    (src_img, sample_img, tgt_img), axis=1), [2, 0, 1]),idx)
                        avg_psnr += Metrics.calculate_psnr(sample_img, tgt_img)
                        avg_ssim += Metrics.calculate_ssim(sample_img, tgt_img)
                        avg_lpips += cal_metrics.cal_lpips(sample_img, tgt_img)


                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}', 
                                np.concatenate((src_img, sample_img, tgt_img), axis=1)
                            )
                    
                    avg_psnr = avg_psnr / idx
                    avg_ssim = avg_ssim / idx
                    avg_lpips = avg_lpips / idx

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train',
                        num_train_timesteps=opt['model']['beta_schedule']['train']['n_timestep'])
                    # log
                    logger.info('# Validation # PSNR: {:.4e}, SSIM:{:.4e}, LPIPS:{:.4e}'.format(avg_psnr,avg_ssim,avg_lpips))

                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e},ssim: {:.4e},lpips: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr,avg_ssim,avg_lpips))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim', avg_ssim, current_step)
                    tb_logger.add_scalar('lpips', avg_lpips, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_ssim': avg_ssim,
                            'validation/val_lpips': avg_lpips,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_lpips = 0.0
        avg_l2 = 0.0

        idx = 0
        result_path = '{}'.format(opt['path']['resume_state'])
        os.makedirs(result_path, exist_ok=True)
        
        if os.path.isdir(os.path.join(result_path, 'sample')):
            exit_filenames = os.listdir(os.path.join(result_path, 'sample'))
        else:
            exit_filenames = []
        
        ttt = 0
        ttti=0
        for _,  val_data in enumerate(val_loader):
            idx += 1
            img_name = val_data['filename'][0]
            if img_name in exit_filenames:
                continue
            
            ttt0 = time.time()
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)    # continous=True 
            ttt+=(time.time()-ttt0)
            ttti +=1

            visuals = diffusion.get_current_visuals()
            sample_img = visuals['SR'][-1]
            inter_samples = visuals['SR'][:-1]
            if len(visuals['SR_onestep']):
                sam_onesteps = visuals['SR_onestep']#[:-1]
                sam_onestep = Metrics.tensor2img(sam_onesteps)  # uint8  
                onestep_dir = os.path.join(result_path, 'sam_onstep')

            os.makedirs(os.path.join(result_path, 'src'), exist_ok=True)
            os.makedirs(os.path.join(result_path, 'tgt'), exist_ok=True)
            os.makedirs(os.path.join(result_path, 'sample'), exist_ok=True)
            
            tgt_img = Metrics.tensor2img(visuals['HR'])  # uint8     
            src_img = Metrics.tensor2img(visuals['LR'])  # uint8 
            sample_img = Metrics.tensor2img(sample_img)  # uint8  

            Metrics.save_img(
                sample_img, os.path.join(result_path, 'sample', img_name))
            
            if len(val_loader)<=100:
                os.makedirs(onestep_dir, exist_ok=True)
                Metrics.save_img(
                            sam_onestep, '{}/{}_{}.png'.format(onestep_dir, current_step, idx))
                Metrics.save_img(
                                src_img, os.path.join(result_path, 'src', img_name))
            
            Metrics.save_img(
                            tgt_img, os.path.join(result_path, 'tgt', img_name))

            inter_samples = torch.cat((inter_samples, visuals['HR']), dim=0)
            inter_samples = Metrics.tensor2img(inter_samples)  # uint8

            eval_psnr = Metrics.calculate_psnr(sample_img, tgt_img)
            eval_ssim = Metrics.calculate_ssim(sample_img, tgt_img)
            eval_lpips = cal_metrics.cal_lpips(sample_img, tgt_img)
            eval_l2 = cal_metrics.cal_l2(sample_img, tgt_img)


            # generation
            avg_psnr += eval_psnr
            avg_ssim += eval_ssim
            avg_lpips += eval_lpips
            avg_l2+=eval_l2

            # if wandb_logger and opt['log_eval']:
            #     wandb_logger.log_eval_data(sample_img, Metrics.tensor2img(visuals['SR'][-1]), tgt_img, eval_psnr, eval_ssim)
        print('avg time:', ttt/ttti)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx
        avg_lpips = avg_lpips / idx
        avg_l2 = avg_l2 / idx

        os.rename(result_path, result_path+'_S%.3f_P%.2f_l2%.3f_Lp%.3f' % (avg_ssim, avg_psnr, avg_l2, avg_lpips))
        
        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger.info('# Validation # LPIPS: {:.4e}'.format(avg_lpips))
        # logger.info('# Validation # FID: {:.4e}'.format(avg_fid))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}, lpips：{:.4e}'.format(
                current_epoch, current_step, avg_psnr, avg_ssim, avg_lpips))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim),
                'LPIPS': float(avg_lpips)
            })
