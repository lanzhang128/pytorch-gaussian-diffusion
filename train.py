import argparse
import json
import os
import logging
import time

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import dataset
import beta_schedule
import time_aware_module
from gaussian_diffusion import GaussianDiffusion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Gaussian diffusion model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('-s', '--seed', default=0, type=int, help='random seed, default: 0')
    args = parser.parse_args()
    
    assert args.config[-4:] == 'json', 'config needs to be a json file!'
    torch.manual_seed(args.seed)
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    args.__dict__.update(config)
    
    model_save_dirpath = args.train_cfg['model_save_dirpath']
    if not os.path.exists(model_save_dirpath):
        os.system('mkdir ' + model_save_dirpath)
    
    with open(os.path.join(model_save_dirpath, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(args.__dict__, f, ensure_ascii=False, indent=4)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_name = os.path.join(args.train_cfg['model_save_dirpath'], time.strftime(
        '%Y%m%d%H%M', time.localtime(time.time())) + '.log')
    file_handler = logging.FileHandler(log_name, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    
    schedule = beta_schedule.__getattribute__(args.schedule['name'])
    schedule = schedule(**args.schedule['kwargs'])
    
    time_aware_net = time_aware_module.__getattribute__(args.time_aware_net['name'])
    time_aware_net = time_aware_net(**args.time_aware_net['kwargs'])
    
    device = 'cuda:0'
    diffusion_model = GaussianDiffusion(schedule, time_aware_net, **args.diffusion)
    diffusion_model.to(device)
    
    lr = args.train_cfg['lr']
    epochs = args.train_cfg['epochs']
    batch_size = args.train_cfg['batch_size']
    
    optimizer = Adam(diffusion_model.parameters(), lr=lr)
    
    Dataset = dataset.__getattribute__(args.dataset['name'])
    train_dataloader = DataLoader(Dataset(**args.dataset['train']), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(Dataset(**args.dataset['valid']), batch_size=batch_size, shuffle=False)
    
    total_step = epochs * len(train_dataloader)
    step_count = 0
    
    val_loss = 1e6
    
    record_period = args.train_cfg['record_period']
    save_period = args.train_cfg['save_period']
    save_best = args.train_cfg['save_best']
    for epoch in range(1, epochs+1):
        temp = '=' * 20 + '  Train  ' + '=' * 20 + f'\nEpoch {epoch}\n' + '=' * 49 + '\nTrain Loss:'
        print(temp)
        logger.debug(temp)
        diffusion_model.train()
        
        total_loss = 0
        start_time = time.time()
        for step, img in enumerate(train_dataloader):
            optimizer.zero_grad()
            img = img.to(device)
            img = img * 2 - 1
            loss = diffusion_model(img)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            step_count += 1
            
            if (step + 1) % record_period == 0:
                est = time.gmtime((time.time() - start_time) * (total_step - step_count) / record_period)
                est = f'{est.tm_yday - 1:>2d}d {est.tm_hour:>2d}h {est.tm_min:>2d}m {est.tm_sec:>2d}s'
                temp = f'{step+1}/{len(train_dataloader)}, loss: {loss.item():<.4f}, estimate time: {est}'
                print(temp)
                logger.debug(temp)
                start_time = time.time()
        total_loss = total_loss / (step + 1)
        print(f'Avg: loss: {total_loss:<.4f}')
        logger.debug(f'Avg: loss: {total_loss:<.4f}')
        
        temp = '=' * 20 + '  Evaluate  ' + '=' * 20
        print(temp)
        logger.debug(temp)
        diffusion_model.eval()
        
        with torch.no_grad():
            total_loss = 0
            for step, img in enumerate(val_dataloader):
                img = img.to(device)
                img = img * 2 - 1
                loss = diffusion_model(img)
                total_loss += loss.item()
                
        total_loss = total_loss / (step + 1)
        print(f'Avg: loss: {total_loss:<.4f}')
        logger.debug(f'Avg: loss: {total_loss:<.4f}')
        
        if epoch % save_period == 0:
            save_path = os.path.join(model_save_dirpath, f'epoch_{epoch}.pth')
            torch.save(diffusion_model.state_dict(), save_path)
            print('Save model at: ' + save_path)
            logger.debug('Save model at: ' + save_path)
        
        if save_best:
            if total_loss < val_loss:
                val_loss = total_loss
                save_path = os.path.join(model_save_dirpath, f'best.pth')
                torch.save(diffusion_model.state_dict(), save_path)
                print('Save model at: ' + save_path)
                logger.debug('Save model at: ' + save_path)
