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
from cub_with_attribute import Cub2011
import training_scripts.evaluator as evaluator
from models.vilt import ViLT_MLM
from training_scripts.logger import log_metrics as logger
from utils.gather import GatherLayer
from utils.distributed import set_distributed_mode
from utils.utils import get_accuracy

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

    val_dataset = Cub2011(val_dataset_args['root'], val_dataset_args, download=download, train = False, normalize = False)
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
        metrics['Class Accuracy 1'] = 0
        metrics['Class Accuracy 2'] = 0
        metrics['Class Accuracy 3'] = 0
        metrics['Class Accuracy 4'] = 0
        metrics['Majority Accuracy'] = 0

    if val:
        val_metrics['total'] = 0
        val_metrics['class total'] = 0
        val_metrics['class total2'] = 0
        val_metrics['Accuracy 1'] = 0
        val_metrics['Accuracy 2'] = 0
        val_metrics['Accuracy 3'] = 0
        val_metrics['Accuracy 4'] = 0
        val_metrics['Class Accuracy 1'] = 0
        val_metrics['Class Accuracy 2'] = 0
        val_metrics['Class Accuracy 3'] = 0
        val_metrics['Class Accuracy 4'] = 0
        val_metrics['Majority Accuracy'] = 0



def return_accuracy(classification, labels, truth):
    majority_classifier = torch.zeros(classification.shape).cuda()

    classification = torch.sigmoid(classification)
    classification = classification > 0.5

    return torch.sum(classification[truth] == labels[truth]), torch.sum(majority_classifier[truth] == labels[truth])

def return_auc(classification, labels, truth, metric):

    return metric.update(torch.sigmoid(classification[truth]), labels[truth])


def create_labeling_vector(attributes, model, images, args):
    labeling_vector = torch.zeros((images.shape[0], model.processor.tokenizer.vocab_size)).cuda()

    #gets the vocab label of the wings and puts it in the labeling vector
    wing_attributes = attributes
    for i, attributes in enumerate(wing_attributes):
        labels = []
        for idx, attribute in enumerate(attributes):
            if attribute == 1:
                wing_color = args.attribute_idx_to_color[idx]
                label = args.color_to_vocab_idx[wing_color]
                labels.append(label)
        amount = 1 / max(1, len(labels))
        for label in labels:
            labeling_vector[i][label] = amount
    return labeling_vector


def create_output_logits(logits, images, args):
    output_logits = torch.zeros((images.shape[0], 15)).cuda()
    for i, row in enumerate(logits):
        for idx, attribute in enumerate(args.color_to_vocab_idx):
            logit_idx = args.color_to_vocab_idx[attribute]
            output_logits[i][idx] = row[logit_idx]
    return output_logits

        
def assess_accuracy(logits, attributes):
    total = 0
    accurate = 0

    for idx, row in enumerate(attributes):
        if row.sum() == 0: continue
        labels = []
        for i, attribute in enumerate(attributes[idx]):
            #no words for iridiscent or rufuous so skip

            if attribute == 1:
                labels.append(i)

        _, top_idx = torch.topk(logits[idx], len(labels))
        top_idx = sorted(top_idx)

        total += len(labels)
        for i in range(len(top_idx)):
            if top_idx[i] in labels:
                accurate += 1
    return accurate, total

def eval_fn(model, classification_number, val_dataloader, args, num_attributes, val_metrics, wandb, step):
    logits_by_att, labels_by_att = defaultdict(list), defaultdict(list)
    
    #print(text_prompts)
    with torch.no_grad():
        for i, batch in enumerate(iter(val_dataloader)):
            image, attributes, certainty = batch['image'].cuda(), batch['attributes'].cuda(), batch['certainty'].cuda()
            #just gets the wings
            prompts = [['The color of the birds wings are [MASK].'] * image.shape[0]][0]
            attributes = attributes[:, 9: 24]
            #0 out iridescent and rufous
            attributes[:, 2] = 0
            attributes[:, 4] = 0

            output = model(prompts, image)
            attributes_logits = create_output_logits(output.logits[:, -1], image, args)
            
            truth = certainty >= args.certainty_threshold
            for att_id in range(num_attributes):
                att_name = att_id
                current_truth = truth[: , att_id]
                
                logits, labels = attributes_logits[current_truth, att_id], attributes[current_truth, att_id]
                logits_by_att[att_name].append(logits)
                labels_by_att[att_name].append(labels)

        acc_list, rocauc_list = [], []
        for key in range(num_attributes):
            #for iridescent and rufous, skip
            if key in [2, 4] : continue
            
            logits, labels = torch.cat(logits_by_att[key], axis=0), torch.cat(labels_by_att[key], axis=0)

            acc = (torch.round(torch.sigmoid(logits)) == labels).type(torch.float32).mean().detach().cpu().numpy()
            rocauc = roc_auc_score(labels.detach().cpu(), logits.detach().cpu())
            acc_list.append(acc), rocauc_list.append(rocauc)
            fpr, tpr, thres = roc_curve(labels.detach().cpu(), logits.detach().cpu())
            
    val_metrics['AUC'] = np.mean(rocauc_list)

    if log and args.rank == 0:
        logger(val_metrics, step, wandb = wandb, train = False, args = args, training_script = False)


def validation_step(data: list, 
                    model: nn.Module, 
                    metrics: dict,
                    step: int,
                    log = False,
                    wandb = None,
                    args = None):
    with torch.no_grad():
        images, attributes = data['image'], data['attributes'].cuda()

        #just gets the wings
        attributes = attributes[:, 9: 24]
        #0 out iridescent and rufous
        attributes[:, 2] = 0
        attributes[:, 4] = 0

        #labeling_vector = create_labeling_vector(attributes, model.module, images, args)

        prompts = [['The color of the birds wings are [MASK].'] * images.shape[0]][0]
        
        output = model(prompts, images)
        output_logits = create_output_logits(output.logits[:, -1], images, args)

        accurate, total = assess_accuracy(output_logits, attributes)    
        
        metrics['total'] += total
        metrics['Accurate'] += accurate
        
        #logging protocol
        if log and args.rank == 0:
            logger(metrics, step, wandb = wandb, train = False, args = args, training_script=True)
            reset_metrics(metrics, val = True)
        
        return None

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Attribute Probing Experiments')
    parser.add_argument('--name', type = str)
    parser.add_argument('--workers', nargs='?', default = 8,  type=int)
    parser.add_argument('--batch_size', nargs='?', default = 128,  type=int)
    parser.add_argument('-log', action='store_true')
    parser.add_argument('--data_path', default = '../../../data/', type = str)
    parser.add_argument('--val_steps', default = 100, type = int)
    parser.add_argument('--saved_path', default=None, type = str)
    parser.add_argument('--certainty_threshold', default = 3, type = int)
    
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
    args.log_n_val_steps = args.val_steps - 1
    
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
        
    model = ViLT_MLM(args=None)
    
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
    val_metrics['Accurate'] = 0

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
    
    now = time.time()
    num_attributes = 15

    eval_fn(model, i, val_dataloader, args, num_attributes, val_metrics, wandb, 0)

    evaluator = evaluator.Evaluator(
                             model,
                             val_dataloader,
                             args, 
                             validation_step,
                             val_metrics,
                             wandb)

    evaluator.evaluate()
    print('Done in', time.time() - now)