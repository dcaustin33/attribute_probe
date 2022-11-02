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
from models.vilt import ViLT_MLM
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
    #not normalizing as the ViLT processor does it for us
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

    #not normalizing as the ViLT processor does it for us
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


def get_params(model: nn.Module, args):
    params = [ {"params": model.parameters(),"weight_decay": 0}]
    return params


def reset_metrics(metrics, val = False):
    if not val:
        metrics['total'] = 0
        metrics['Accurate'] = 0
        metrics['Total Loss'] = 0
        metrics['CE Loss'] = 0

    if val:
        metrics['total'] = 0
        metrics['Accurate'] = 0

def create_labeling_vector(attributes, model, images, args):
    labeling_vector = torch.zeros((images.shape[0], model.processor.tokenizer.vocab_size)).cuda()

    #gets the vocab label of the wings and puts it in the labeling vector
    wing_attributes = attributes[11:26]
    for i, attributes in enumerate(wing_attributes):
        labels = []
        for idx, attribute in enumerate(attributes):
            if attribute == 1:
                wing_color = args.attribute_idx_to_color[idx]
                label = args.color_to_vocab_idx[wing_color]
                labels.append(label)
        amount = 1 / len(labels)
        for label in labels:
            labeling_vector[i][label] = amount
    return labeling_vector

def assess_accuracy(logits, labeling_vector, attributes):
    total = 0
    accurate = 0

    for idx, row in labeling_vector:
        if row.sum() == 0: continue
        labels = []
        for i, attribute in enumerate(attributes[idx]):
            if attribute == 1:
                labels.append(i)
        correct_idx = [args.attribute_idx_to_color[i] for i in labels]
        correct_vocab_idx = [args.color_to_vocab_idx[i] for i in correct_idx]

        _, top_idx = torch.topk(logits[idx], len(labels))
        top_idx = sorted(top_idx)
        correct_vocab_idx = sorted(correct_vocab_idx)

        total += len(labels)
        for i in range(len(labels)):
            if top_idx[i] != correct_vocab_idx[i]:
                accurate += 1

    return accurate, total






def training_step(data: dict, 
               model: nn.Module, 
               metrics: dict,
               step: int,
               log = False,
               wandb = None,
               args = None) -> torch.Tensor:
    
    images, attributes, certainty, class_labels = data['image'], data['attributes'].cuda(), data['certainty'].cuda(), data['class'].cuda()

    truth = certainty >= args.certainty_threshold

    labeling_vector = create_labeling_vector(attributes, model, images, args)

    prompts = [['The color of the birds wings are [MASK]'] * images.shape[0]]
    
    output = model(prompts, images)

    loss = F.cross_entropy(output.logits, labeling_vector)

    accurate, total = assess_accuracy(output.logits, labeling_vector, attributes)    
    
    metrics['total'] += total
    metrics['accurate'] += accurate
    metrics['Total Loss'] += loss
    metrics['CE Loss'] += loss
    
    #logging protocol
    if log and args.rank == 0:
        logger(metrics, step, wandb = wandb, train = True, args = args, training_script=True)
        reset_metrics(metrics, val = False)
    
    return loss

    
def validation_step(data: list, 
                    model: nn.Module, 
                    metrics: dict,
                    step: int,
                    log = False,
                    wandb = None,
                    args = None):
    with torch.no_grad():
        images, attributes, certainty, class_labels = data['image'], data['attributes'].cuda(), data['certainty'].cuda(), data['class'].cuda()

        truth = certainty >= args.certainty_threshold

        labeling_vector = create_labeling_vector(attributes, model, images, args)

        prompts = [['The color of the birds wings are [MASK]'] * images.shape[0]]
        
        output = model(prompts, images)

        accurate, total = assess_accuracy(output.logits, labeling_vector, attributes)    
        
        metrics['total'] += total
        metrics['accurate'] += accurate
        
        #logging protocol
        if log and args.rank == 0:
            logger(metrics, step, wandb = wandb, train = True, args = args, training_script=True)
            reset_metrics(metrics, val = False)
        
        return None

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Attribute Probing Experiments')
    parser.add_argument('--name', type = str)
    parser.add_argument('--lr', nargs='?', default = .0001, type=float)
    parser.add_argument('--workers', nargs='?', default = 8,  type=int)
    parser.add_argument('--steps', nargs='?', default = 8000,  type=int)
    parser.add_argument('--batch_size', nargs='?', default = 128,  type=int)
    parser.add_argument('--val_steps', nargs='?', default = 70,  type=int)
    parser.add_argument('--log_n_steps', nargs='?', default = 800,  type=int)
    parser.add_argument('-log', action='store_true')
    parser.add_argument('--data_path', default = '../../../data/', type = str)
    parser.add_argument('--log_n_train_steps', default = 100, type = int)
    parser.add_argument('-checkpoint', action='store_true')
    parser.add_argument('--checkpoint_path', default = None, type = str)
    parser.add_argument('--certainty_threshold', default = 3, type = int)
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

    #makes so the directory is readable in the dataset
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
    
    args.wd = 1.5e-6
    args.steps += 1
    args.log_n_val_steps = args.val_steps - 1
    args.warmup_steps = 200
    args.classifier_lr = args.lr
    args.classifier_weight_decacy = 0


    set_distributed_mode(args)  
    log = args.log
    name = args.name
    
    dataset, dataloader, val_dataloader = prepare_data(args.dataset_args, args.val_dataset_args)
    args.epochs = args.steps // dataset.__len__()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Working on device", device)
    
    #creates a logger otherwise makes null logger
    if log and args.rank == 0:
        wandb = wandb.init(config = args, name = name, project = 'attribute_probe')
    else: wandb = None

    metrics = {}
    metrics['total'] = 0
    metrics['Accurate'] = 0
    metrics['Total Loss'] = 0
    metrics['CE Loss'] = 0
    
    val_metrics = {}
    val_metrics['total'] = 0
    val_metrics['Accurate'] = 0


    model = ViLT_MLM(args=None)
    #loads from a checkpoint
    if args.checkpoint:
        checkpoint = torch.load('{name}'.format(name = args.checkpoint_path))
        new_dict = {}
        for i in checkpoint['model_state_dict']:
            new_dict[i[7:]] = checkpoint['model_state_dict'][i]
        model.load_state_dict(new_dict)
        model = model.cuda()
        
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        params = get_params(model, args)
        optimizer = torch.optim.AdamW(params, lr = args.lr, weight_decay = args.wd)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        schedule = LinearWarmupCosineAnnealingLR(
                        optimizer,
                        warmup_epochs= args.warmup_steps,
                        max_epochs= args.steps,
                        warmup_start_lr=3e-05,
                        eta_min=0,
                        last_epoch= checkpoint['step'])
        steps = checkpoint['step']
        
        print('Restarting from step:', steps, 'with learning rate', schedule.get_last_lr()[0])
    else:
        steps = 0
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        params = get_params(model, args)
        optimizer = torch.optim.AdamW(params, lr = args.lr, weight_decay = args.wd)
        schedule = LinearWarmupCosineAnnealingLR(
                            optimizer,
                            warmup_epochs= args.warmup_steps,
                            max_epochs= args.steps,
                            warmup_start_lr=3e-05,
                            eta_min=0)


    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu_id])

    os.environ["TOKENIZERS_PARALLELISM"] = "false" 

    #give the model the idx of the wings
    args.color_to_vocab_idx = {'blue': 2630, 
                        'brown': 2829, 
                        'purple': 6379, 
                        'grey': 4462, 
                        'yellow': 3756, 
                        'olive': 9724, 
                        'green': 2665, 
                        'pink': 5061, 
                        'orange': 4589, 
                        'black': 2304, 
                        'white': 2317, 
                        'red': 2417, 
                        'buff': 23176}
    args.mlm_idx_to_collect = sorted([args.color_to_vocab_idx[i] for i in args.color_to_vocab_idx])

    args.color_to_attribute_idx = {'blue': 0,
                                    'brown': 1,
                                    'iridescent': 2,
                                    'purple': 3,
                                    'rufous': 4,
                                    'grey': 5,
                                    'yellow': 6,
                                    'olive': 7,
                                    'green': 8,
                                    'pink': 9,
                                    'orange': 10,
                                    'black': 11,
                                    'white':12 ,
                                    'red': 13,
                                    'buff': 14}
    #flip the dictionary
    args.attribute_idx_to_color = {v: k for k, v in args.color_to_attribute_idx.items()}
    
    trainer = trainer.Trainer(
                             model,
                             dataloader, 
                             val_dataloader,
                             args, 
                             training_step,
                             validation_step,
                             optimizer, 
                             schedule, 
                             current_step = steps,
                             metrics = metrics,
                             val_metrics = val_metrics,
                             wandb = wandb)
    
    trainer.train()