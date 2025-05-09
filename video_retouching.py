# -*- coding: utf-8 -*-
import importlib
import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from core.dataset_video import VideoDataset_test
import torch.nn.functional as F
from torchvision.utils import save_image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description="STTN") #
parser.add_argument("-e", "--epoch", type=str, default="")
parser.add_argument("--size",  type=int, default=512)
parser.add_argument("-c", "--ckpt", type=str, default= "release_model")
parser.add_argument("--model", type=str, default='VRetouchEr')
parser.add_argument("--input_path", type=str, default="datasets/video")  # wild image
parser.add_argument("--save_path", type=str, default= "results/video")
args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    print(args.model)
    data = torch.load("{0}/gen_{1}.pth".format(args.ckpt, args.epoch), map_location=device)
    model.load_state_dict(data)
    model.eval()
    test_dataset = VideoDataset_test(path = args.input_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    save_path = args.save_path
    if not os.path.exists(save_path) or len(os.listdir(save_path))<7:
        os.makedirs(os.path.join(save_path), exist_ok=True)
        print("make dirs")
    print(save_path)
    for name, source_tensor_list in tqdm(test_loader):
        name = name[0]
        with torch.no_grad():
            list = [source_tensor_list[-1].to(device)]
            for i in range(6):
                source_tensor_list[i] = source_tensor_list[i].to(device)
            pred_img, atten, _ = model(source_tensor_list)
            path = os.path.join(args.save_path, "show_1", f"{str(name)}.png")
            save_image(F.interpolate(pred_img, size=512), path, normalize=True, value_range=(-1, 1))
