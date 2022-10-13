import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils import data
import os
import torch.distributed as dist

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_accuracy(predictions, labels):
    if len(predictions) > 0:
        _, predicted = torch.max(predictions, 1)
        acc1 = (predicted == labels).sum()

        _, pred = predictions.topk(5)
        labels = labels.unsqueeze(1).expand_as(pred)
        acc5 = (labels == pred).any(dim = 1).sum()
        return acc1, acc5
    return 0, 0

def get_accuracy_concat(nouns, adjectives, label_nouns, label_adjectives):
    if len(nouns) > 0:
        _, predicted_noun = torch.max(nouns, 1)
        _, predicted_adjective = torch.max(adjectives, 1)
        acc1_noun = (predicted_noun == label_nouns)
        acc1_adj = (predicted_adjective == label_adjectives)
        acc1 = (acc1_noun & acc1_adj).sum()

        _, pred_noun = nouns.topk(5)
        _, pred_adj = adjectives.topk(5)
        label_nouns = label_nouns.unsqueeze(1).expand_as(pred_noun)
        label_adjectives = label_adjectives.unsqueeze(1).expand_as(pred_adj)
        acc5_noun = (label_nouns == pred_noun).any(dim = 1)
        acc5_adj = (label_adjectives == pred_adj).any(dim = 1)
        acc5 = (acc5_noun & acc5_adj).sum()
        return acc1, acc5
    return 0, 0