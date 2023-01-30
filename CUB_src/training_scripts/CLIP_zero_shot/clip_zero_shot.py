#torch inputs
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils import data
from torch.multiprocessing import Pool, Process, set_start_method
import torch.distributed as dist
import torchmetrics

#imports defined in folder
from cub_dataset import Cub2011
import training_scripts.trainer as trainer
import torchvision
from models.clip_text_image_pretrained import CLIP_text_image
from training_scripts.logger import log_metrics as logger
from utils.gather import GatherLayer
from utils.distributed import set_distributed_mode
from utils.utils import get_accuracy

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
    train_dataset = Cub2011(dataset_args['root'], dataset_args, download=download)
    sampler = data.DistributedSampler(train_dataset)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers if args.workers else len(os.sched_getaffinity(0)),
        pin_memory=False,
        drop_last=True,
        persistent_workers=True
    )

    val_dataset = Cub2011(dataset_args['root'], val_dataset_args, download=download, train = False)
    val_sampler = data.DistributedSampler(val_dataset)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers if args.workers else len(os.sched_getaffinity(0)),
        pin_memory=False,
        drop_last=True,
        persistent_workers=True
    )
    return train_dataset, dataloader, val_dataloader

def training_step(data: dict, 
               model: nn.Module, 
               metrics: dict,
               step: int,
               log = False,
               wandb = None,
               args = None) -> torch.Tensor:
    
    images, attributes, certainty, class_labels = data['image'].cuda(), data['attributes'].cuda(), data['certainty'].cuda(), data['class'].cuda()

    truth = certainty >= args.certainty_threshold
    classification_out, clip_image_logits = model(images)


    #adding second term to loss function so gradients are not zero
    loss = clip_image_logits.sum() * 0
    accuracys = []
    for i in classification_out[:4]: 
        loss += cross_entropy_loss(i, attributes, truth)
        accuracys.append(return_accuracy(i, attributes, truth)[0])
    _, maj_accuracy = return_accuracy(i, attributes, truth)
    for i in range(1, len(accuracys) + 1):
        metrics['Accuracy {}'.format(i)] += accuracys[i - 1]

    accuracys = []

    for i in classification_out[4:]:
        loss += F.cross_entropy(i, class_labels)
        accuracys.append(get_accuracy(i, class_labels))
    for i in range(1, len(accuracys) + 1):
        metrics['Class Accuracy {}'.format(i)] += accuracys[i - 1][0]

    
    
    metrics['total'] += torch.sum(truth)
    metrics['class total'] += images.shape[0]
    metrics['Total Loss'] += loss
    metrics['CE Loss'] += loss
    metrics['Majority Accuracy'] += maj_accuracy
    
    #logging protocol
    if log and args.rank == 0:
        logger(metrics, step, wandb = wandb, train = True, args = args, training_script=True)
        reset_metrics(metrics, val = False)
    
    return loss


    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Attribute Probing Experiments')
    parser.add_argument('--name', type = str)
    parser.add_argument('--workers', nargs='?', default = 8,  type=int)
    parser.add_argument('--batch_size', nargs='?', default = 128,  type=int)
    parser.add_argument('--data_path', default = '../../../data/', type = str)
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
                 'crop_size': 224,
                 'brightness': 0.4, 
                 'contrast': 0.4, 
                 'saturation': .2, 
                 'hue': .1, 
                 'color_jitter_prob': 0, 
                 'gray_scale_prob': 0, 
                 'horizontal_flip_prob': 0.5, 
                 'gaussian_prob': .5, 
                 'min_scale': 0.99, 
                 'max_scale': 1.0}
    
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
                 'gaussian_prob': .5, 
                 'min_scale': 0.99, 
                 'max_scale': 1.0}

    set_distributed_mode(args)  

    dataset, dataloader, val_dataloader = prepare_data(args.dataset_args, args.val_dataset_args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Working on device", device)
        
    model = CLIP_text_image(args=None)

    model = model.cuda()
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 

    text_prompt = ['A photo of a bird with a red belly']

    text_embed = model.forward_text(text_prompt).cpu().detach()
    #create an empty tensor to store the image embeddings
    image_embeddings = torch.empty((0, 512), dtype=torch.float32)
    indicies = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images, attributes, certainty, class_labels = data['image'].cuda(), data['attributes'].cuda(), data['certainty'].cuda(), data['class'].cuda()
            new_image_embed = model.forward_image(images)
            new_image_embed = new_image_embed.detach().cpu()
            #stack the image_embeddings
            image_embeddings = torch.cat((image_embeddings, new_image_embed), 0)
            #append each individual part of idx tensor
            for i in data['idx']: indicies.append(i.item())
        for i, data in enumerate(val_dataloader):
            images, attributes, certainty, class_labels = data['image'].cuda(), data['attributes'].cuda(), data['certainty'].cuda(), data['class'].cuda()
            new_image_embed = model.forward_image(images)
            new_image_embed = new_image_embed.detach().cpu()
            #stack the image_embeddings
            image_embeddings = torch.cat((image_embeddings, new_image_embed), 0)
            #append each individual part of idx tensor
            for i in data['idx']: indicies.append(i.item()+ 30000)
        
        #calculate the cosine similarity between the text and image embeddings
        similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine_similarity = similarity(text_embed, image_embeddings)
        values, idx = torch.topk(cosine_similarity, 100)
        with open('readouts/red_belly_cosine_similarity.txt', 'w') as f:
            for i in range(len(values)):
                f.write(str(indicies[idx[i]]) + ' ' + str(values[i].item()) + '\n')
    