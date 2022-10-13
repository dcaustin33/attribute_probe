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

#imports defined in folder
from mit_states_dataset import MIT_states
import training_scripts.trainer as trainer
import torchvision
from models.clip_text_image_pretrained import CLIP_text_image, CLIP_text_image_concat
from training_scripts.logger import log_metrics as logger
from utils.gather import GatherLayer
from utils.distributed import set_distributed_mode
from utils.utils import get_accuracy, get_accuracy_concat

#python helper inputs
import os
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import wandb
import pytorch_lightning as pl
import time


torch.backends.cudnn.enabled = True 
torch.backends.cudnn.benchmark = True


#prepare the data for SimCLR
def prepare_data(dataset_args, val_dataset_args):
    download = False
    
    #create the dataset
    train_dataset = MIT_states(dataset_args['root'], dataset_args, download=download)
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

    val_dataset = MIT_states(dataset_args['root'], val_dataset_args, download=download, train = False)
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


def get_params(model: nn.Module, 
               args):
    params = [  
                {"params": model.linear_adjectives.parameters(), "lr": args.classifier_lr, "weight_decay": 0},
                {"params": model.linear_nouns.parameters(), "lr": args.classifier_lr, "weight_decay": 0},
                {"params": model.classifier_adjectives.parameters(), "lr": args.classifier_lr, "weight_decay": 0},
                {"params": model.classifier_nouns.parameters(), "lr": args.classifier_lr, "weight_decay": 0},
                ]
    return params


def cross_entropy_loss(classification, labels, truth):
    return F.cross_entropy(classification[truth], labels[truth])


def return_accuracy(classification, labels, truth):
    return get_accuracy(classification[truth], labels[truth])


def reset_metrics(metrics, val = False):
    if not val:
        metrics['total'] = 0
        metrics['class total'] = 0
        metrics['Total Loss'] = 0
        metrics['CE Loss'] = 0
        metrics['Linear Adjective Accuracy'] = 0
        metrics['Linear Noun Accuracy'] = 0
        metrics['Linear Concat Accuracy'] = 0
        metrics['Classifier Adjective Accuracy'] = 0
        metrics['Classifier Noun Accuracy'] = 0
        metrics['Classifier Concat Accuracy'] = 0

    if val:
        val_metrics['total'] = 0
        val_metrics['class total'] = 0
        val_metrics['class total2'] = 0
        val_metrics['Linear Adjective Accuracy'] = 0
        val_metrics['Linear Noun Accuracy'] = 0
        val_metrics['Linear Concat Accuracy'] = 0
        val_metrics['Classifier Adjective Accuracy'] = 0
        val_metrics['Classifier Noun Accuracy'] = 0
        val_metrics['Classifier Concat Accuracy'] = 0

def transform_images(images):
    for i in range(images.shape[0]):
        images[i] = dataset.transform(images[i].cuda())
    return images




def training_step(data: dict, 
               model: nn.Module, 
               metrics: dict,
               step: int,
               log = False,
               wandb = None,
               args = None) -> torch.Tensor:
    
    images, adjective_labels, noun_labels, concat_labels = data['image'].cuda(), data['adjective_labels'].cuda(), data['noun_labels'].cuda(), data['concat_labels'].cuda()
    labels = [adjective_labels, noun_labels] * 2

    truth = data['noun_labels'] != -1
    classification_out, clip_image_logits = model(images)


    #adding second term to loss function so gradients are not zero
    loss = clip_image_logits.sum() * 0
    accuracys = []
    for idx, i in enumerate(classification_out): 
        loss += cross_entropy_loss(i, labels[idx], truth)
        accuracys.append(return_accuracy(i, labels[idx], truth)[0])
    accuracys.append(get_accuracy_concat(classification_out[0], classification_out[1], labels[0], labels[1])[0])
    accuracys.append(get_accuracy_concat(classification_out[2], classification_out[3], labels[0], labels[1])[0])

    
    metrics['Linear Adjective Accuracy'] += accuracys[0]
    metrics['Linear Noun Accuracy'] += accuracys[1]
    metrics['Classifier Adjective Accuracy'] += accuracys[2]
    metrics['Classifier Noun Accuracy'] += accuracys[3]
    metrics['Linear Arg Max Concat Accuracy'] += accuracys[4]
    metrics['Classifier Arg Max Concat Accuracy'] += accuracys[5]

    metrics['total'] += torch.sum(truth)
    metrics['class total'] += torch.sum(truth)
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
        images, adjective_labels, noun_labels, concat_labels = data['image'].cuda(), data['adjective_labels'].cuda(), data['noun_labels'].cuda(), data['concat_labels'].cuda()
        labels = [adjective_labels, noun_labels] * 2

        truth = data['noun_labels'] != -1
        classification_out, clip_image_logits = model(images)


        #adding second term to loss function so gradients are not zero
        loss = clip_image_logits.sum() * 0
        accuracys = []
        for idx, i in enumerate(classification_out): 
            accuracys.append(return_accuracy(i, labels[idx], truth)[0])
        accuracys.append(get_accuracy_concat(classification_out[0], classification_out[1], labels[0], labels[1])[0])
        accuracys.append(get_accuracy_concat(classification_out[2], classification_out[3], labels[0], labels[1])[0])

        
        metrics['Linear Adjective Accuracy'] += accuracys[0]
        metrics['Linear Noun Accuracy'] += accuracys[1]
        metrics['Classifier Adjective Accuracy'] += accuracys[2]
        metrics['Classifier Noun Accuracy'] += accuracys[3]
        metrics['Linear Arg Max Concat Accuracy'] += accuracys[4]
        metrics['Classifier Arg Max Concat Accuracy'] += accuracys[5]

        metrics['total'] += torch.sum(truth)
        metrics['class total'] += torch.sum(truth)
        
        #logging protocol
        if log and args.rank == 0:
            logger(metrics, step, wandb = wandb, train =False, args = args, training_script=True)
            reset_metrics(metrics, val = True)
        
        return loss

def create_text_prompts_adjectives(adjectives):
    text_prompts = []
    for word in adjectives:
        text_prompts.append('This photo shows an object that is ' + word)
    return text_prompts

def create_text_prompts_nouns(nouns):
    text_prompts = []
    for word in nouns:
        text_prompts.append('This is a ' + word)
    return text_prompts


    
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
                 'transformation': False,
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
                 'transformation': False,
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
    
    
    if log and args.rank == 0:
        wandb = wandb.init(config = args, name = name, project = 'attribute_probe')
    else: wandb = None
        
    adjectives = len(dataset.adjectives.keys())
    nouns = len(dataset.nouns.keys())
    concat = len(dataset.concat.keys())

    model = CLIP_text_image_concat(adjectives = adjectives, nouns = nouns, concat = concat, args=None)

    metrics = {}
    metrics['total'] = 0
    metrics['class total'] = 0
    metrics['Total Loss'] = 0
    metrics['CE Loss'] = 0
    metrics['Linear Adjective Accuracy'] = 0
    metrics['Linear Noun Accuracy'] = 0
    metrics['Linear Concat Accuracy'] = 0
    metrics['Classifier Adjective Accuracy'] = 0
    metrics['Classifier Noun Accuracy'] = 0
    metrics['Classifier Concat Accuracy'] = 0
    metrics['Linear Arg Max Concat Accuracy'] = 0
    metrics['Classifier Arg Max Concat Accuracy'] = 0
    

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
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_id]
    )
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 

    adjectives = []
    for i in dataset.adjectives:
        adjectives.append(dataset.adjectives[i])
    nouns = []
    for i in dataset.nouns:
        nouns.append(dataset.nouns[i])

    adjective_text_prompts = create_text_prompts_adjectives(adjectives)
    noun_text_prompts = create_text_prompts_nouns(nouns)

    model.module.create_text_embeddings(adjective_text_prompts, noun_text_prompts)
    
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