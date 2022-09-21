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
from cub_dataset import Cub2011
import training_scripts.evaluator as evaluator
from models.clip_text_image_pretrained import CLIP_text_image
from training_scripts.logger import log_metrics as logger
from utils.gather import GatherLayer
from utils.distributed import set_distributed_mode

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

    val_dataset = Cub2011(val_dataset_args['root'], val_dataset_args, download=download, train = False)
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
    majority_classifier = torch.zeros(classification.shape).cuda()

    classification = torch.sigmoid(classification)
    classification = classification > 0.5

    return torch.sum(classification[truth] == labels[truth]), torch.sum(majority_classifier[truth] == labels[truth])

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


def eval_fn(model, classification_number, val_dataloader, args, num_attributes, val_metrics, wandb, step):
    logits_by_att, labels_by_att = defaultdict(list), defaultdict(list)
    prompts = create_text_prompts(args)
    with torch.no_grad():
        for i, batch in enumerate(iter(val_dataloader)):
            image, attributes, certainty = batch['image'].cuda(), batch['attributes'].cuda(), batch['certainty'].cuda()
            attributes_logits, _ = model(prompts, image)
            attributes_logits = attributes_logits[classification_number]
            truth = certainty >= 3
            for att_id in range(num_attributes):
                att_name = att_id
                current_truth = truth[: , att_id]
                
                logits, labels = attributes_logits[current_truth, att_id], attributes[current_truth, att_id]
                logits_by_att[att_name].append(logits)
                labels_by_att[att_name].append(labels)

        acc_list, rocauc_list = [], []
        for key in range(num_attributes):
            
            logits, labels = torch.cat(logits_by_att[key], axis=0), torch.cat(labels_by_att[key], axis=0)

            acc = (torch.round(torch.sigmoid(logits)) == labels).type(torch.float32).mean().detach().cpu().numpy()
            rocauc = roc_auc_score(labels.detach().cpu(), logits.detach().cpu())
            acc_list.append(acc), rocauc_list.append(rocauc)
            fpr, tpr, thres = roc_curve(labels.detach().cpu(), logits.detach().cpu())
            
    val_metrics['AUC {}'.format(classification_number + 1)] = np.mean(rocauc_list)
    val_metrics['Accuracy {}'.format(classification_number + 1)] = np.mean(acc_list)

    if log and args.rank == 0:
        logger(val_metrics, step, wandb = wandb, train = False, args = args, training_script = False)

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Attribute Probing Experiments')
    parser.add_argument('--name', type = str)
    parser.add_argument('--workers', nargs='?', default = 8,  type=int)
    parser.add_argument('--batch_size', nargs='?', default = 128,  type=int)
    parser.add_argument('-log', action='store_true')
    parser.add_argument('--data_path', default = '../../../data/', type = str)
    parser.add_argument('--log_n_train_steps', default = 100, type = int)
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
    
    dataset, val_dataloader = prepare_data(args.val_dataset_args)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Working on device", device)
    
    
    if log and args.rank == 0:
        wandb = wandb.init(config = args, name = name, project = 'attribute_probe')
    else: wandb = None
        
    model = CLIP_text_image(args=None)
    
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
    
    now = time.time()
    
    #testing each classification network
    for i in range(2):
        eval_fn(model, i, val_dataloader, args, dataset.num_attributes, val_metrics, wandb, 0)
    print('Done in', time.time() - now)