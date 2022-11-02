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
from models.clip_text_image_pretrained import CLIP_text_image_with_attribute
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


def create_text_prompts(args):
    with open(args.data_path + '/CUB_200_2011/attributes.txt', 'r') as f:
        lines = f.readlines()
    text_prompts = []
    for line in lines:

        #base that will be used for every image
        start = ''

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
    text_prompts = np.array(create_text_prompts(args))
    #print(text_prompts)
    with torch.no_grad():
        for i, batch in enumerate(iter(val_dataloader)):
            image, attributes, certainty = batch['image'].cuda(), batch['attributes'].cuda(), batch['certainty'].cuda()
            prompt = 'The bird has a '
            if args.attribute_idx_amount > 1:
                arr = text_prompts[batch['attribute_idx']]
                
                correct_prompts = [prompt + ', '.join(i) + '.' for i in list(arr)]
            else:
                arr = text_prompts[batch['attribute_idx']]
                correct_prompts = [prompt + i + '.' for i in list(arr)]
            attributes_logits, _ = model(correct_prompts, image)
            attributes_logits = attributes_logits[classification_number]
            truth = certainty >= args.certainty_threshold
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

def eval_fn_with_attribute(model, classification_number, val_dataloader, args, num_attributes, val_metrics, wandb, step):
    logits_by_att, labels_by_att = [[], []], [[], []]
    text_prompts = np.array(create_text_prompts(args))
    #print(text_prompts)
    with torch.no_grad():
        for i, batch in enumerate(iter(val_dataloader)):
            print(i)
            image, attributes, certainty = batch['image'].cuda(), batch['attributes'].cuda(), batch['certainty'].cuda()
            prompt = 'The bird has a '
            if args.attribute_idx_amount > 1:
                arr = text_prompts[batch['attribute_idx']]
                correct_prompts = [prompt + ', '.join(i) + '.' for i in list(arr)]
            else:
                arr = text_prompts[batch['attribute_idx']]
                correct_prompts = [prompt + i + '.' for i in list(arr)]
            attributes_logits, _ = model(correct_prompts, image)
            attributes_logits = attributes_logits[classification_number]
            truth = certainty >= args.certainty_threshold

            for im_id in range(len(image)):
                if args.attribute_idx_amount > 1:
                    att_id = np.random.choice(args.attribute_idx_amount)
                    att_id = batch['attribute_idx'][im_id][att_id]
                else: att_id = batch['attribute_idx'][im_id]
                att_id2 = np.random.choice(np.where(batch['attributes'][im_id] == 1)[0], size = 1)[0]
                chosen = batch['attribute_idx'][im_id].numpy()
                if args.attribute_idx_amount == 1: chosen = [chosen]
                while att_id2 in chosen:
                    att_id2 = np.random.choice(np.where(batch['attributes'][im_id] == 1)[0], size = 1)
                
                logit, label = attributes_logits[im_id, att_id], attributes[im_id, att_id]
                logits_by_att[0].append(logit)
                labels_by_att[0].append(label)

                logit, label = attributes_logits[im_id, att_id2], attributes[im_id, att_id2]
                logits_by_att[1].append(logit)
                labels_by_att[1].append(label)

        #print(logits_by_att[0])
        acc_list, acc_list2 = [], []
            
        logits, labels = torch.tensor(logits_by_att[0]), torch.tensor(labels_by_att[0])
        logits2, labels2 = torch.tensor(logits_by_att[1]), torch.tensor(labels_by_att[1])

        acc = (torch.round(torch.sigmoid(logits)) == labels).type(torch.float32).mean().detach().cpu().numpy()
        acc_list.append(acc)
        acc2 = (torch.round(torch.sigmoid(logits2)) == labels2).type(torch.float32).mean().detach().cpu().numpy()
        acc_list2.append(acc2)


    val_metrics['Specific Attribute Accuracy {}'.format(classification_number + 1)] = np.mean(acc_list)
    val_metrics['Random Present Attribute Accuracy {}'.format(classification_number + 1)] = np.mean(acc_list2)

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
        images, attributes, certainty, class_labels = data['image'].cuda(), data['attributes'].cuda(), data['certainty'].cuda(), data['class'].cuda()

        truth = certainty >= args.certainty_threshold

        #give the text prompts with a real attribute for each
        prompt = 'The bird has a '
        if args.attribute_idx_amount > 1:
            arr = text_prompts[data['attribute_idx']]
            
            correct_prompts = [prompt + ', '.join(i) + '.' for i in list(arr)]
        else:
            arr = text_prompts[data['attribute_idx']]
            correct_prompts = [prompt + i + '.' for i in list(arr)]
        classification_out, clip_image_logits = model(correct_prompts, images)

        accuracys = []

        for i in classification_out[4:]:
            accuracys.append(get_accuracy(i, class_labels))
        for i in range(1, len(accuracys) + 1):
            metrics['Class Accuracy {}'.format(i)] += accuracys[i - 1][0]

        
        
        metrics['total'] += torch.sum(truth)
        metrics['class total'] += images.shape[0]
        
        #logging protocol
        if log and args.rank == 0:
            logger(metrics, step, wandb = wandb, train = False, args = args, training_script=True)
        
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
    args.log_n_val_steps = args.val_steps - 1
    
    args.val_dataset_args = {
                 'root': args.data_path,
                 'attribute_idx_amount': args.attribute_idx_amount,
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
    print(dataset.__len__())
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Working on device", device)
    
    
    if log and args.rank == 0:
        wandb = wandb.init(config = args, name = name, project = 'attribute_probe')
    else: wandb = None
        
    model = CLIP_text_image_with_attribute(args=None)
    
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
    val_metrics['Class Accuracy 1'] = 0
    val_metrics['Class Accuracy 2'] = 0
    val_metrics['Class Accuracy 3'] = 0
    val_metrics['Class Accuracy 4'] = 0
    
    now = time.time()
    
    #testing each classification network
    for i in range(4):
        print(i)
        eval_fn_with_attribute(model, i, val_dataloader, args, dataset.num_attributes, val_metrics, wandb, 0)
    for i in range(4):
        print(i)
        eval_fn(model, i, val_dataloader, args, dataset.num_attributes, val_metrics, wandb, 0)

    evaluator = evaluator.Evaluator(
                             model,
                             val_dataloader,
                             args, 
                             validation_step,
                             val_metrics,
                             wandb)
    text_prompts = np.array(create_text_prompts(args))
    evaluator.evaluate()
    print('Done in', time.time() - now)