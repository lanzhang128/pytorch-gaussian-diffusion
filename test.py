import argparse
import json
import os
from PIL import Image

import torch
import numpy as np

import beta_schedule
import time_aware_module
from gaussian_diffusion import GaussianDiffusion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a Gaussian diffusion model')
    parser.add_argument('ckpt', help='checkpoint file path')
    parser.add_argument('shape', type=int, nargs='+', help='shape of image')
    parser.add_argument('-s', '--seed', default=0, type=int, help='random seed, default: 0')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    with open(os.path.join(os.path.dirname(args.ckpt), 'args.json'), 'r') as f:
        config = json.load(f)
    args.__dict__.update(config)

    schedule = beta_schedule.__getattribute__(args.schedule['name'])
    schedule = schedule(**args.schedule['kwargs'])
    
    time_aware_net = time_aware_module.__getattribute__(args.time_aware_net['name'])
    time_aware_net = time_aware_net(**args.time_aware_net['kwargs'])
    
    device = 'cuda:0'
    diffusion_model = GaussianDiffusion(schedule, time_aware_net, **args.diffusion)
    diffusion_model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    diffusion_model.to(device)
    diffusion_model.eval()
    
    with torch.no_grad():
        if not os.path.exists(f'{args.ckpt[:-4]}_gen'):
            os.system(f'mkdir {args.ckpt[:-4]}_gen')
            
        xT = diffusion_model.sample_p_xT((10, 3, args.shape[0], args.shape[1]), device)
        bin = [i*100 for i in range(10, -1, -1)]
    
        x = xT
        res = [(x.clip(-1, 1).permute(0, 2, 3, 1).cpu().numpy() + 1) / 2 * 255]
        for i in range(len(bin)-1):
            x = diffusion_model.generate(x, bin[i], bin[i+1])
            res.append((x.clip(-1, 1).permute(0, 2, 3, 1).cpu().numpy() + 1) / 2 * 255)
        
    for j in range(10):
        img = []
        for i in range(len(res)):
            img.append(res[i][j].astype('uint8'))
        image = Image.fromarray(np.hstack(img))
        image.save(f'{args.ckpt[:-4]}_gen/{j}.png')
            