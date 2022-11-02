import torchmetrics
import torch
from sklearn.metrics import roc_auc_score, roc_curve

def log_metrics(metrics: dict, 
                step: int,
                args,
                wandb = None, 
                train = True,
                training_script = True
                ) -> None:

    if train:
        print(step, "Loss:", round(metrics['Total Loss'].item(), 2))
    if training_script:
        for i in metrics:
            if 'Accuracy' in i and 'Class' not in i:
                metrics[i] = metrics[i] / metrics['total']
            if 'Accuracy' in i and 'Class' in i:
                metrics[i] = metrics[i] / metrics['class total']
            if 'Accurate' in i:
                metrics[i] = metrics[i] / metrics['total']

    print('In Logging', wandb, args.rank)
    if wandb and args.rank == 0:
        if not train:
            new_metrics = {}
            for i in metrics:
                if 'Accuracy' in i or 'AUC' in i:
                    new_metrics['Val ' + i] = metrics[i]
            print('logging')
            wandb.log(new_metrics, step = step)
        else:
            wandb.log(metrics, step = step)

    return