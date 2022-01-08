import torch
import torch.nn as nn
from dataloader import MapDataset
from torch.utils.data import DataLoader
from discriminator import PatchGAN
from generator import UNet
from dataloader import MapDataset
import time
from torchvision import transforms
import os
from tqdm import tqdm
import mmcv
import argparse
from torch import optim
import matplotlib.pyplot as plt
from glob import glob
from utils import tensor2image, generate_animation

def initialize_weights(model) :
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1 :
        nn.init.normal_(model.weight.data, 0.0, 0.02)

# configuration file
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default = './config.yaml', help = 'configuration path for pix2pix')
args = parser.parse_args()


cfg = mmcv.Config.fromfile(args.config_file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), transforms.Resize((256,256))])

# model
model_dis = PatchGAN(in_channels = cfg.in_channels)
model_gen = UNet(in_channels = cfg.in_channels, out_channels = cfg.out_channels)


model_dis.apply(initialize_weights)
model_gen.apply(initialize_weights)

if cfg.weight_path :
    if os.path.isfile(os.path.join(cfg.weight_path, 'weights_dis.pt')) :
        model_dis.load_state_dict(torch.load(os.path.join(cfg.weight_path, 'weights_dis.pt')))
        print('Load Discriminator model weight!')
    if os.path.isfile(os.path.join(cfg.weight_path, 'weights_gen.pt')) :
        model_gen.load_state_dict(torch.load(os.path.join(cfg.weight_path, 'weights_gen.pt')))
        print('Load Generator model weight!')


model_dis.to(device)
model_gen.to(device)

# data loader
dataset = MapDataset(data_dir = cfg.train_data_dir, transform = transform)
train_dataloader = DataLoader(dataset, batch_size = cfg.batch_size, shuffle = True, drop_last = True)

# loss function
loss_func_gan = nn.BCELoss()
loss_func_pix = nn.L1Loss()

# optimizer
opt_dis = optim.Adam(model_dis.parameters(), lr = cfg.lr, betas = (cfg.beta1, cfg.beta2))
opt_gen = optim.Adam(model_gen.parameters(), lr = cfg.lr, betas = (cfg.beta1, cfg.beta2))


# train
model_dis.train()
model_gen.train()

batch_count = 0

start_time = time.time()
loss_hist = {'gen' : [],
            'dis' : []}

if cfg.val_data_dir :
    val_dataset = MapDataset(data_dir = cfg.val_data_dir, transform = transform, train = False)
    val_dataloader = DataLoader(val_dataset, batch_size = val_dataset.__len__())


os.makedirs(cfg.weight_path, exist_ok = True)
os.makedirs(cfg.res_path, exist_ok = True)

for epoch in tqdm(range(cfg.num_epochs)) :
    for real_img_batch, map_img_batch in train_dataloader :
        batch_size = real_img_batch.size(0)
        
        # real
        real_img_batch = real_img_batch.to(device)
        map_img_batch = map_img_batch.to(device)
        
        # patch label
        real_label = torch.ones(cfg.batch_size, *(1, cfg.patch_num, cfg.patch_num), requires_grad = False).to(device)
        fake_label = torch.zeros(cfg.batch_size, *(1, cfg.patch_num, cfg.patch_num), requires_grad = False).to(device)
        
        # generator
        model_gen.zero_grad()
        
        fake_map_img_batch = model_gen(real_img_batch)
        out_dis = model_dis(fake_map_img_batch, map_img_batch)
        
        gen_loss = loss_func_gan(out_dis, real_label)
        pixel_loss = loss_func_pix(fake_map_img_batch, map_img_batch)
        
        g_loss = gen_loss + cfg.lambda_pixel * pixel_loss
        g_loss.backward()
        opt_gen.step()
        
        # discriminator
        model_dis.zero_grad()
        
        out_dis = model_dis(map_img_batch, real_img_batch)
        real_loss = loss_func_gan(out_dis, real_label)
        
        out_dis = model_dis(fake_map_img_batch.detach(), real_img_batch)
        fake_loss = loss_func_gan(out_dis, fake_label)
        
        d_loss = (real_loss + fake_loss) / 2.
        d_loss.backward()
        opt_dis.step()
        
        loss_hist['gen'].append(g_loss.item())
        loss_hist['dis'].append(d_loss.item())
        

    print('Epoch : %.0f, G_loss : %.6f, D_loss : %.6f, time: %.2f min' %(epoch, g_loss.item(), d_loss.item(), (time.time() - start_time) / 60 ))
    
    # validation step
    if cfg.val_data_dir :
        test_imgs = next(iter(val_dataloader)).to(device)
        result = model_gen(test_imgs)
        for res, imgname in zip(result, val_dataloader.dataset.img_list) :
            placename = os.path.splitext(os.path.basename(imgname))[0]
            os.makedirs(os.path.join(cfg.res_path, placename), exist_ok = True)
            fname = os.path.join(cfg.res_path, placename, placename + '_epoch%03d.jpg'%(epoch+1))
            plt.imsave(fname, tensor2image(res)*0.5 + 0.5) #
        
    
    
    torch.save(model_gen.state_dict(), os.path.join(cfg.weight_path, 'weights_gen.pt'))
    torch.save(model_dis.state_dict(), os.path.join(cfg.weight_path, 'weights_dis.pt'))
                

# animation
res_folders = glob(os.path.join(cfg.res_path, '*'))
for res_folder in res_folders : 
    if os.path.isdir(res_folder) :
        path = os.path.join(res_folder, res_folder.split('/')[-1])
        generate_animation(path = path, num = cfg.num_epochs)
        