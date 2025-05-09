import os
import glob
import logging
import importlib
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from core.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from core.loss import GANLoss, VGGLoss, IDLoss, AttentionLoss_video, SSIM
from core.dataset_video import FaceRetouchingDataset_video_new
from model.modules.flow_comp import BlemishFlow_Loss
import wandb
import lpips
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from model.discriminator import MultiscaleDiscriminator
from torch.optim.lr_scheduler import OneCycleLR
class Trainer:
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        self.maxiteration = 0
        self.train_dataset = FaceRetouchingDataset_video_new(path = config['train_data_loader']['dataroot'],
                                                    resolution=700, data_type="train", get_mask=False,
                                                    data_percentage=config['train_data_loader']['percentage'], frame_num=self.config['model']['frame_num'])
        self.test_dataset = FaceRetouchingDataset_video_new(path = "datasets/ffhqr_hard",
                                                    resolution=512, data_type="test", get_mask=False,
                                                    data_percentage=1, frame_num=self.config['model']['frame_num'])
        self.frame_num = self.config['model']['frame_num']
        print("net:", self.config['model']['net'])
        if config['trainer']['use_wandb'] == 1:
            wandb.init(project="retouching", name=self.config['model']['net'] + str(self.frame_num) +"_138")
            self.wandb = True       
        else:

            self.wandb = False
        self.train_sampler = None
        self.train_args = config['trainer']
        if config['distributed']:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'],
                rank=config['global_rank'])

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None),
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=4)
        self.eval_txt = config['eval_txt']
        # set loss functions
        # self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS'])
        # self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.adversarial_loss = GANLoss(gan_mode=self.config['losses']['GAN_LOSS'])
        self.l1_loss = nn.SmoothL1Loss().to(self.config['device'])
        self.lpips_loss = lpips.LPIPS(net='alex', lpips=False).to(self.config['device'])
        self.vgg_loss = VGGLoss(device=self.config['device'])
        self.attention_loss = AttentionLoss_video(device=self.config['device'])
        self.ID_loss = IDLoss(device=self.config['device'])
        self.ssim_loss = SSIM(window_size=11)
        self.BlemishFLow_loss = BlemishFlow_Loss()
        # setup models including generator and discriminator
        net = importlib.import_module('model.' + config['model']['net'])
        self.netG = net.InpaintGenerator(n_layer_t=self.config['model']['n_layer_t'], frame_num=self.config['model']['frame_num'])
        # print(self.netG)
        self.netG = self.netG.to(self.config['device'])
        if not self.config['model']['no_dis']:
            # self.netD = net.Discriminator(
            #     in_channels=3,
            #     use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
            # self.netD = self.netD.to(self.config['device'])
            self.netD = MultiscaleDiscriminator(num_D=self.config['model']['num_D'], n_layers_D=self.config['model']['n_layers_D'])
            self.netD = self.netD.to(self.config['device'])
        # print(self.netD)
        # setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.load()
        self.eval_mask(self.iteration)

        if config['distributed']:
            self.netG = DDP(self.netG,
                            device_ids=[self.config['local_rank']],
                            output_device=self.config['local_rank'],
                            broadcast_buffers=True,
                            find_unused_parameters=True)
            if not self.config['model']['no_dis']:
                self.netD = DDP(self.netD,
                                device_ids=[self.config['local_rank']],
                                output_device=self.config['local_rank'],
                                broadcast_buffers=True,
                                find_unused_parameters=False)

        # set summary writer
        # self.dis_writer = None
        # self.gen_writer = None
        # self.summary = {}
        # if self.config['global_rank'] == 0 or (not config['distributed']):
        #     self.dis_writer = SummaryWriter(
        #         os.path.join(config['save_dir'], 'dis'))
        #     self.gen_writer = SummaryWriter(
        #         os.path.join(config['save_dir'], 'gen'))

    def setup_optimizers(self):
        """Set up optimizers."""
        backbone_params = []
        for name, param in self.netG.named_parameters():
            backbone_params.append(param)

        optim_params = [
            {
                'params': backbone_params,
                'lr': self.config['trainer']['lr']
            }
        ]

        self.optimG = torch.optim.Adam(optim_params,
                                       betas=(self.config['trainer']['beta1'],
                                              self.config['trainer']['beta2']))

        if not self.config['model']['no_dis']:
            self.optimD = torch.optim.Adam(
                self.netD.parameters(),
                lr=self.config['trainer']['lr'],
                betas=(self.config['trainer']['beta1'],
                       self.config['trainer']['beta2']))        

    def setup_schedulers(self):
        """Set up schedulers."""
        scheduler_opt = self.config['trainer']['scheduler']
        # scheduler_type = scheduler_opt.pop('type')

        # if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
        #     self.scheG = MultiStepRestartLR(self.optimG, milestones=scheduler_opt['milestones'],
        #                                     gamma=scheduler_opt['gamma'])
        #     self.scheD = MultiStepRestartLR(self.optimD, milestones=scheduler_opt['milestones'],
        #                                     gamma=scheduler_opt['gamma'])
        # elif scheduler_type == 'CosineAnnealingRestartLR':
        self.scheG = CosineAnnealingRestartLR(
                self.optimG,
                periods=scheduler_opt['periods'],
                restart_weights=scheduler_opt['restart_weights'])
        self.scheD = CosineAnnealingRestartLR(
                self.optimD,
                periods=scheduler_opt['periods'],
                restart_weights=scheduler_opt['restart_weights'])
        # self.scheG = OneCycleLR(
        #     self.optimG,
        #     max_lr=self.config["trainer"]["lr"],
        #     total_steps=200000,
        #     pct_start=0.025,
        #     final_div_factor=2e3,
        #     div_factor=10
        # )
        # self.scheD = OneCycleLR(
        #     self.optimG,
        #     max_lr=self.config["trainer"]["lr"],
        #     total_steps=200000,
        #     pct_start=0.025,
        #     final_div_factor=2e3,
        #     div_factor=10
        # )
        # elif scheduler_type == "ExponentialLR":
        #     self.scheG = ExponentialLR(self.optimG, gamma=0.7)
        #     self.scheD = ExponentialLR(self.optimD, gamma=0.7)
        # else:
        #     raise NotImplementedError(
        #         f'Scheduler {scheduler_type} is not implemented yet.')

    def update_learning_rate(self):
        """Update learning rate."""
        self.scheG.step()
        self.scheD.step()

    def get_lr(self):
        """Get current learning rate."""
        # return self.optimG.param_groups[0]['lr']
        return self.scheG.get_lr()[0]

    # def add_summary(self, writer, name, val):
    #     """Add tensorboard summary."""
    #     if name not in self.summary:
    #         self.summary[name] = 0
    #     self.summary[name] += val
    #     if writer is not None and self.iteration % 100 == 0:
    #         writer.add_scalar(name, self.summary[name] / 100, self.iteration)
    #         self.summary[name] = 0

    def load(self):
        """Load netG (and netD)."""
        # get the latest checkpoint
        model_path = self.config['save_dir']
        # data = torch.load("model/modules/conv.pth", map_location=self.config['device'])
        # self.netG.conv.load_state_dict(data)
        # data = torch.load("model/modules/vrt.pth", map_location=self.config['device'])
        # self.netG.vrt.load_state_dict(data)
        # data = torch.load("model/modules/blemish.pth", map_location=self.config['device'])
        # self.netG.BlemishLSTM.load_state_dict(data)
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            ckpts = [
                os.path.basename(i).split('.pth')[0]
                for i in glob.glob(os.path.join(model_path, '*.pth'))
            ]
            ckpts.sort()
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None

        if latest_epoch is not None:
            gen_path = os.path.join(model_path,
                                    f'gen_{int(latest_epoch):06d}.pth')
            dis_path = os.path.join(model_path,
                                    f'dis_{int(latest_epoch):06d}.pth')
            opt_path = os.path.join(model_path,
                                    f'opt_{int(latest_epoch):06d}.pth')

            if self.config['global_rank'] == 0:
                print(f'Loading model from {gen_path}...')
            dataG = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(dataG)
            if not self.config['model']['no_dis']:
                dataD = torch.load(dis_path,
                                   map_location=self.config['device'])
                self.netD.load_state_dict(dataD)

            data_opt = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data_opt['optimG'])
            self.scheG.load_state_dict(data_opt['scheG'])
            if not self.config['model']['no_dis']:
                self.optimD.load_state_dict(data_opt['optimD'])
                self.scheD.load_state_dict(data_opt['scheD'])
            self.epoch = data_opt['epoch']
            self.iteration = data_opt['iteration']
            self.test(self.iteration, lr = self.get_lr())
        else:
            if self.config['global_rank'] == 0:
                print('Warnning: There is no trained model found. An initialized model will be used.')

    def save(self, it):
        """Save parameters every eval_epoch"""
        if self.config['global_rank'] == 0:
            # configure path
            gen_path = os.path.join(self.config['save_dir'],
                                    f'gen_{it:06d}.pth')
            dis_path = os.path.join(self.config['save_dir'],
                                    f'dis_{it:06d}.pth')
            opt_path = os.path.join(self.config['save_dir'],
                                    f'opt_{it:06d}.pth')
            print(f'\nsaving model to {gen_path} ...')

            # remove .module for saving
            if isinstance(self.netG, torch.nn.DataParallel) \
               or isinstance(self.netG, DDP):
                netG = self.netG.module
                if not self.config['model']['no_dis']:
                    netD = self.netD.module
            else:
                netG = self.netG
                if not self.config['model']['no_dis']:
                    netD = self.netD

            # save checkpoints
            torch.save(netG.state_dict(), gen_path)
            if not self.config['model']['no_dis']:
                torch.save(netD.state_dict(), dis_path)
                torch.save(
                    {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optimD': self.optimD.state_dict(),
                        'scheG': self.scheG.state_dict(),
                        'scheD': self.scheD.state_dict()
                    }, opt_path)
            else:
                torch.save(
                    {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'scheG': self.scheG.state_dict()
                    }, opt_path)

            latest_path = os.path.join(self.config['save_dir'], 'latest.ckpt')
            os.system(f"echo {it:06d} > {latest_path}")
    
    def test(self, iteration, lr):
        self.netG.eval()
        cnt = 0
        PSNR = 0
        SSIM = 0
        LPIPS = 0
        device = self.config['device']
        loss_fn = lpips.LPIPS(net='alex').to(device)
        for name, source_tensor_list, target_tensor_list in tqdm(self.test_loader):
            with torch.no_grad(): 
                for i in range(len(source_tensor_list)):
                    source_tensor_list[i] = source_tensor_list[i].to(device)
                pred_img, _, _ = self.netG(source_tensor_list)
                lpips_loss = loss_fn(pred_img, target_tensor_list[-1].to(device)).mean()
                s_img = pred_img[0].cpu().numpy()
                t_img = target_tensor_list[-1][0].numpy()
                psnr = compare_psnr(t_img, s_img)
                ssim = compare_ssim(t_img, s_img, channel_axis=0, data_range=t_img.max() - t_img.min())
                PSNR += psnr
                SSIM += ssim
                LPIPS+= lpips_loss
                cnt+=1
        PSNR/=cnt
        SSIM/=cnt
        LPIPS/=cnt
        print(iteration, ": PSNR:",PSNR, "SSIM:", SSIM, "LPIPS:", LPIPS)
        if self.wandb:
            wandb.log({"PSNR" :PSNR.item(),"SSIM" :SSIM.item(),"LPIPS" :LPIPS.item()})
        with open(self.eval_txt,'a') as f:
            f.writelines(f"lr: {lr}; {iteration}: PSNR: {PSNR}; SSIM: {SSIM}; LPIPS: {LPIPS}\n")  
        f.close()
        self.netG.train()

    def eval_mask(self, iteration):
        self.netG.eval()
        trans = transforms.Compose([transforms.Resize(512), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        source_tensor = trans(Image.open("./datasets/retouching/test/source/63000.png").convert("RGB")).unsqueeze(0).cuda()
        source_list = []
        for i in range(self.frame_num):
            source_list.append(source_tensor)
        with torch.no_grad():
            pred_img, atten, _ = self.netG(source_list)
            atten = torch.cat((atten), dim=0)
            atten = (atten - atten.amin(dim=[2, 3], keepdim=True)) / (atten.amax(dim=[2, 3], keepdim=True) - atten.amin(dim=[2, 3], keepdim=True))
            save_image(atten, f'results/check/blemish_{iteration}.png', normalize=True, value_range=(-1, 1), nrow=6)
            m = torch.cat((source_tensor, pred_img), dim=0)
            save_image(m, f'results/check/result_{iteration}.png', normalize=True, value_range=(-1, 1), nrow=2)
        self.netG.train()

    def train(self):
        """training entry"""
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar,
                        initial=self.iteration,
                        dynamic_ncols=True,
                        smoothing=0.01)

        os.makedirs('logs', exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(filename)s[line:%(lineno)d]"
            "%(levelname)s %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
            filename=f"logs/{self.config['save_dir'].split('/')[-1]}.log",
            filemode='w')
        
        self.optimG.zero_grad()
        self.optimG.step()
        self.optimD.zero_grad()
        self.optimD.step()
        
        while True:
            self.epoch += 1      
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)
            self._train_epoch(pbar)    
            # self.update_learning_rate()
            if self.iteration > self.train_args['iterations']:
                break
        print('\nEnd training....')

    
    def pair(self, source_tensor_list, target_tensor_list):
        b, c, h, w = source_tensor_list[-1].size()         
        pred_imgs, mask, blemish_flow = self.netG(source_tensor_list)
        pred_imgs = pred_imgs.view(b, c, h, w)

        gen_loss = 0

        
        real_clip = self.netD(target_tensor_list[-1])
        fake_clip = self.netD(pred_imgs.detach())
        dis_real_loss = self.adversarial_loss(real_clip, True, True)
        dis_fake_loss = self.adversarial_loss(fake_clip, False, True)
        dis_loss = (dis_real_loss + dis_fake_loss) / 2
        # self.add_summary(self.dis_writer, 'loss/dis_vid_fake', dis_fake_loss.item())
        # self.add_summary(self.dis_writer, 'loss/dis_vid_real', dis_real_loss.item())
        self.optimD.zero_grad()
        dis_loss.backward()
        self.optimD.step()

        # generator adversarial loss
        gen_clip = self.netD(pred_imgs)
        gan_loss = self.adversarial_loss(gen_clip, True, False)
        gan_loss = gan_loss * self.config['losses']['adversarial_weight']
        gen_loss += gan_loss
        # self.add_summary(self.gen_writer, 'loss/gan_loss', gan_loss.item())

        # generator l1 loss
        valid_loss = self.l1_loss(pred_imgs, target_tensor_list[-1])
        valid_loss = valid_loss * self.config['losses']['valid_weight']
        gen_loss += valid_loss
        # self.add_summary(self.gen_writer, 'loss/valid_loss', valid_loss.item())

        # mask attention loss
        mask_atten_loss = self.attention_loss(mask, source_tensor_list, target_tensor_list)
        mask_atten_loss = mask_atten_loss * self.config['losses']['mask_weight']
        gen_loss += mask_atten_loss
        # self.add_summary(self.gen_writer, 'loss/mask_atten_loss', mask_atten_loss.item())
            
        # VGG loss
        vgg_loss = self.vgg_loss(pred_imgs, target_tensor_list[-1])
        vgg_loss = vgg_loss * self.config['losses']['vgg_weight']
        gen_loss += vgg_loss
        # self.add_summary(self.gen_writer, 'loss/vgg_loss', vgg_loss.item())
            
        # ID loss
        id_loss = self.ID_loss(pred_imgs, target_tensor_list[-1], source_tensor_list[-1])
        id_loss = id_loss * self.config['losses']['id_weight']
        gen_loss += id_loss
        # self.add_summary(self.gen_writer, 'loss/id_loss', id_loss.item())

        # lpips loss
        lpips_loss = self.lpips_loss.forward(pred_imgs, target_tensor_list[-1]).mean()
        lpips_loss = lpips_loss * self.config['losses']['lpips_weight']
        gen_loss += lpips_loss
        # self.add_summary(self.gen_writer, 'loss/lpips_loss', lpips_loss.item())
        
        # ssim loss
        ssim_loss = 1 - self.ssim_loss(pred_imgs, target_tensor_list[-1])
        ssim_loss = ssim_loss * self.config['losses']['ssim_weight']
        gen_loss += ssim_loss
        # self.add_summary(self.gen_writer, 'loss/ssim_loss', ssim_loss.item())

        # blemish flow loss
        flow_loss = self.BlemishFLow_loss(blemish_flow, source_tensor_list, target_tensor_list)
        flow_loss = flow_loss * self.config['losses']['flow_weight']
        gen_loss += flow_loss

        self.optimG.zero_grad()
        gen_loss.backward()
        self.optimG.step()            
        return dis_loss, valid_loss, id_loss, lpips_loss, mask_atten_loss, ssim_loss, gan_loss, flow_loss, vgg_loss

    def _train_epoch(self, pbar):
        """Process input and calculate loss every training epoch"""
        device = self.config['device']
        for name, source_tensor_list, target_tensor_list in self.train_loader:
            # m = torch.cat(source_tensor_list, dim=0)
            # save_image(m, f"/home/ma-user/work/retouching/datasets/{name[0]}_source.png", normalize=True, value_range=(-1, 1))
            # m = torch.cat(target_tensor_list, dim=0)
            # save_image(m, f"/home/ma-user/work/retouching/datasets/{name[0]}_target.png", normalize=True, value_range=(-1, 1))
            for i in range(len(source_tensor_list)):
                source_tensor_list[i] = source_tensor_list[i].to(device)
                target_tensor_list[i] = target_tensor_list[i].to(device)
            self.iteration += 1
            dis_loss, valid_loss, id_loss, lpips_loss, mask_atten_loss, ssim_loss, gan_loss, flow_loss, vgg_loss = \
                self.pair(source_tensor_list, target_tensor_list)
            # if self.iteration < 2e5:
            #     self.scheG.step()
            if self.iteration % 15e3 == 0:
                self.scheG.step()
                self.scheD.step()
            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                lr = self.get_lr()
                pbar.set_description((f"d: {dis_loss.item():.3f}; "
                                      f"valid: {valid_loss.item():.3f}; "
                                      # f"vgg_loss: {vgg_loss.item():.3f}; "
                                      # f"id_loss: {id_loss.item():.3f}; "
                                      # f"lpips_loss: {lpips_loss.item():.3f}; "
                                      f"ssim: {ssim_loss.item():.3f}; "
                                      # f"gan: {gan_loss.item():.3f}; "
                                      # f"unpair_loss: {unpair_loss.item():.6f}; "
                                      f"mask_atten_loss: {mask_atten_loss.item():.3f}; "
                                      f"flow_loss: {flow_loss.item():.3f}; "
                                      f"lr: {lr:.6f}"
                                     ))
                
                if self.wandb:
                    wandb.log({
                    "dis" :dis_loss.item(),
                    "valid":valid_loss.item(),
                    "vgg_loss":vgg_loss.item(),
                    "lpips_loss":lpips_loss.item(),
                    "mask_atten_loss": mask_atten_loss.item(),
                    "ssim": ssim_loss.item(),
                    "id_loss": id_loss.item(),
                    "gan": gan_loss.item(),
                    "flow_loss": flow_loss.item(),
                    "lr": lr
                })
            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration))
                self.test(self.iteration, lr=self.get_lr())
                # self.eval_mask(self.iteration)
                if self.iteration == 2e4:
                    self.test(self.iteration, lr = self.get_lr())
                if self.iteration == 6e4:
                    self.test(self.iteration, lr = self.get_lr())
                # if self.iteration == 12e4:
                #     self.test(self.iteration, lr = self.get_lr())
                if self.iteration > 12e4-1:
                    self.test(self.iteration, lr = self.get_lr())
            if self.iteration > self.train_args['iterations']:
                break