#torch inputs
from cgitb import text
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils import data
from torch.multiprocessing import Pool, Process, set_start_method
import torch.distributed as dist
import torchmetrics
from utils.utils import get_accuracy

#imports defined in folder
from cub_with_attribute import Cub2011
import training_scripts.trainer as trainer
import torchvision
from models.vilt import ViLT_itc
from training_scripts.logger import log_metrics as logger
from utils.gather import GatherLayer
from utils.distributed import set_distributed_mode

#python helper inputs
import os
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import wandb
import pytorch_lightning as pl
import time
import numpy as np


torch.backends.cudnn.enabled = True 
torch.backends.cudnn.benchmark = True


#prepare the data for SimCLR
def prepare_data(dataset_args, val_dataset_args):
    download = False
    
    #create the dataset
    train_dataset = Cub2011(dataset_args['root'], dataset_args, download=download, normalize = False)
    sampler = data.DistributedSampler(train_dataset)
    dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers if args.workers else len(os.sched_getaffinity(0)),
        pin_memory=False,
        drop_last=True,
        persistent_workers=True
    )

    val_dataset = Cub2011(dataset_args['root'], val_dataset_args, download=download, train = False, normalize = False)
    val_sampler = data.DistributedSampler(val_dataset)
    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=args.batch_size,
        num_workers=args.workers if args.workers else len(os.sched_getaffinity(0)),
        pin_memory=False,
        drop_last=True,
        persistent_workers=True
    )
    return train_dataset, dataloader, val_dataloader


    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Attribute Probing Experiments')
    parser.add_argument('--name', type = str)
    parser.add_argument('--workers', nargs='?', default = 8,  type=int)
    parser.add_argument('--batch_size', nargs='?', default = 128,  type=int)
    parser.add_argument('-log', action='store_true')
    parser.add_argument('--data_path', default = '../../../data/', type = str)
    parser.add_argument('--log_n_train_steps', default = 100, type = int)
    parser.add_argument('--attribute_idx_amount', default = 1, type = int,
                    help="""This is how many correct attributes to use as the prompt. Ex 2 would mean use Wing Color Blue and Pointy Beak""")

    
    #distributed arguments
    parser.add_argument("--dist_url", default="tcp://localhost:40000", type=str,
                    help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=-1, type=int, 
                    help="""number of processes: it is set automatically and should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int, 
                    help="""rank of this process: it is set automatically and should not be passed as argument""")
    parser.add_argument("--job_id", default=0, type=int, 
                    help="""rank of this process: it is set automatically and should not be passed as argument""")
    args = parser.parse_args()
    if args.data_path[-1] != '/': args.data_path = args.data_path + '/'

    args.dataset_args = {
                 'root': args.data_path,
                 'attribute_idx_amount': args.attribute_idx_amount,
                 'crop_size': 224,
                 'brightness': 0.4, 
                 'contrast': 0.4, 
                 'saturation': .2, 
                 'hue': .1, 
                 'color_jitter_prob': .4, 
                 'gray_scale_prob': 0.2, 
                 'horizontal_flip_prob': 0.5, 
                 'gaussian_prob': .5, 
                 'min_scale': 0.6, 
                 'max_scale': 0.95}
    
    args.val_dataset_args = {
                 'root': args.data_path,
                 'crop_size': 224,
                 'attribute_idx_amount': args.attribute_idx_amount,
                 'brightness': 0.4, 
                 'contrast': 0.4, 
                 'saturation': .2, 
                 'hue': .1, 
                 'color_jitter_prob': 0, 
                 'gray_scale_prob': 0, 
                 'horizontal_flip_prob': 0.5, 
                 'gaussian_prob': 0, 
                 'min_scale': 0.9, 
                 'max_scale': 1}


    set_distributed_mode(args)  
    dataset, dataloader, val_dataloader = prepare_data(args.dataset_args, args.val_dataset_args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Working on device", device)
    
    model = ViLT_itc(args=None)


    model = model.cuda()
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    
    text_prompt = [['A photo of a bird with a red belly'] * args.batch_size][0]

    #create an empty tensor to store the image embeddings as floats
    similarity = torch.empty(0).float()
    indicies = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images, attributes, certainty, class_labels = data['image'].cuda(), data['attributes'].cuda(), data['certainty'].cuda(), data['class'].cuda()
            output = model(text_prompt, images)
            similarity = torch.cat((similarity, output.logits[:, 0].cpu()), 0)
            for i in data['idx']: indicies.append(i.item())

        #calculate the cosine similarity between the text and image embeddings

        values, idx = torch.topk(similarity, 100)
        with open('readouts/red_belly_cosine_similarity.txt', 'w') as f:
            for i in range(len(values)):
                f.write(str(indicies[idx[i]]) + ' ' + str(values[i].item()) + '\n')
    