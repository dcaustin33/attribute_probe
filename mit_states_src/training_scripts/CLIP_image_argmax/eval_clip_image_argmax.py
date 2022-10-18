#torch inputs
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils import data
from torch.multiprocessing import Pool, Process, set_start_method
import torch.distributed as dist
import torchmetrics
from sklearn.metrics import roc_auc_score, roc_curve

#imports defined in folder
from mit_states_dataset import MIT_states
import training_scripts.evaluator as evaluator
from models.clip_image_pretrained import CLIP_image
from training_scripts.logger import log_metrics as logger
from utils.gather import GatherLayer
from utils.distributed import set_distributed_mode
from utils.utils import get_accuracy, get_accuracy_concat
#python helper inputs
import os
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import wandb
import pytorch_lightning as pl
from collections import defaultdict
import numpy as np
import time


torch.backends.cudnn.enabled = True 
torch.backends.cudnn.benchmark = True


#prepare the data for SimCLR
def prepare_data(val_dataset_args):
    download = False
    
    #create the dataset
    train_dataset = MIT_states(val_dataset_args['root'], val_dataset_args, download=download)

    val_dataset = MIT_states(val_dataset_args['root'], val_dataset_args, download=download, train = False)
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
    return val_dataset, val_dataloader


def return_accuracy(classification, labels, truth):
    return get_accuracy(classification[truth], labels[truth])

def return_auc(classification, labels, truth, metric):

    return metric.update(torch.sigmoid(classification[truth]), labels[truth])

def reset_metrics(metrics, val = False):
    if not val:
        metrics['total'] = 0
        metrics['class total'] = 0
        metrics['Total Loss'] = 0
        metrics['CE Loss'] = 0
        metrics['Accuracy 1'] = 0
        metrics['Accuracy 2'] = 0
        metrics['Accuracy 3'] = 0
        metrics['Accuracy 4'] = 0
        metrics['Majority Accuracy'] = 0

    if val:
        val_metrics['total'] = 0
        val_metrics['class total'] = 0
        val_metrics['class total2'] = 0
        val_metrics['Accuracy 1'] = 0
        val_metrics['Accuracy 2'] = 0
        val_metrics['Accuracy 3'] = 0
        val_metrics['Accuracy 4'] = 0
        val_metrics['Majority Accuracy'] = 0


def create_text_prompts(args):
    with open(args.data_path + '/CUB_200_2011/attributes.txt', 'r') as f:
        lines = f.readlines()
    text_prompts = []
    for line in lines:

        #base that will be used for every image
        start = 'The bird has a '

        #get the words before seeing the descriptor
        beginning = ''
        seen = False


        for i in line.split()[1].split('_'):

            #:: signigifies that the attribute value is on the other side
            if '::' in i:
                first_half = i.split('::')[0]
                second_half = i.split('::')[1]
                seen = True

            #if we have seen the descriptor, we are done and ( signifies that
            if '(' in i:
                break
            if i != 'has':
                if '::' in i: continue
                if seen: second_half += ' ' + i
                else: beginning += i + ' '
        start += second_half + ' ' + beginning  + first_half
        text_prompts.append(start)
    return text_prompts


def validation_step(data: list, 
                    model: nn.Module, 
                    metrics: dict,
                    step: int,
                    log = False,
                    wandb = None,
                    args = None):

    with torch.no_grad():
        images, adjective_labels, noun_labels, concat_labels = data['image'].cuda(), data['adjective_labels'].cuda(), data['noun_labels'].cuda(), data['concat_labels'].cuda()
        if not args.val_dataset_args['transformation']:
            images = transform_images(images)
        labels = [adjective_labels, noun_labels, concat_labels] * 2

        truth = data['noun_labels'] != -1
        classification_out, clip_image_logits = model(images)


        #adding second term to loss function so gradients are not zero
        loss = clip_image_logits.sum() * 0
        accuracys = []
        for idx, i in enumerate(classification_out): 
            accuracys.append(return_accuracy(i, labels[idx], truth)[0])
        accuracys.append(get_accuracy_concat(classification_out[0], classification_out[1], labels[0], labels[1])[0])
        accuracys.append(get_accuracy_concat(classification_out[3], classification_out[4], labels[0], labels[1])[0])

        
        metrics['Linear Adjective Accuracy'] += accuracys[0]
        metrics['Linear Noun Accuracy'] += accuracys[1]
        metrics['Linear Concat Accuracy'] += accuracys[2]
        metrics['Classifier Adjective Accuracy'] += accuracys[3]
        metrics['Classifier Noun Accuracy'] += accuracys[4]
        metrics['Classifier Concat Accuracy'] += accuracys[5]
        metrics['Linear Arg Max Concat Accuracy'] += accuracys[6]
        metrics['Classifier Arg Max Concat Accuracy'] += accuracys[7]

        metrics['total'] += torch.sum(truth)
        metrics['class total'] += torch.sum(truth)
        
        #logging protocol
        if log and args.rank == 0:
            logger(metrics, step, wandb = wandb, train = False, args = args, training_script=True)
            reset_metrics(metrics, val = True)
        
        return loss

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Attribute Probing Experiments')
    parser.add_argument('--name', type = str)
    parser.add_argument('--workers', nargs='?', default = 8,  type=int)
    parser.add_argument('--batch_size', nargs='?', default = 128,  type=int)
    parser.add_argument('-log', action='store_true')
    parser.add_argument('--data_path', default = '../../../data/', type = str)
    parser.add_argument('--log_n_val_steps', default = 100, type = int)
    parser.add_argument('--val_steps', default = 100, type = int)
    parser.add_argument('--saved_path', default=None, type = str)
    
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
    
    args.val_dataset_args = {
                 'root': args.data_path,
                 'transformation': True,
                 'crop_size': 224,
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

    log = args.log
    name = args.name
    args.log_n_val_steps = args.val_steps - 1
    
    dataset, val_dataloader = prepare_data(args.val_dataset_args)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Working on device", device)
    
    
    if log and args.rank == 0:
        wandb = wandb.init(config = args, name = name, project = 'attribute_probe')
    else: wandb = None
        
    adjectives = len(dataset.adjectives.keys())
    nouns = len(dataset.nouns.keys())
    concat = len(dataset.concat.keys())

    model = CLIP_image(adjectives = adjectives, nouns = nouns, concat = concat, args=None)
    
    checkpoint = torch.load('{name}'.format(name = args.saved_path))
    new_dict = {}
    for i in checkpoint['model_state_dict']:
        new_dict[i[7:]] = checkpoint['model_state_dict'][i]
    model.load_state_dict(new_dict)
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_id]
    )

    val_metrics = {}
    val_metrics['total'] = 0
    val_metrics['class total'] = 0
    val_metrics['class total2'] = 0
    val_metrics['Linear Adjective Accuracy'] = 0
    val_metrics['Linear Noun Accuracy'] = 0
    val_metrics['Linear Concat Accuracy'] = 0
    val_metrics['Classifier Adjective Accuracy'] = 0
    val_metrics['Classifier Noun Accuracy'] = 0
    val_metrics['Classifier Concat Accuracy'] = 0
    val_metrics['Linear Arg Max Concat Accuracy'] = 0
    val_metrics['Classifier Arg Max Concat Accuracy'] = 0
    now = time.time()
    
    evaluator = evaluator.Evaluator(
                             model,
                             val_dataloader,
                             args, 
                             validation_step,
                             val_metrics,
                             wandb)
    
    evaluator.evaluate()