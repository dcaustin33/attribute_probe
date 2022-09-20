import wandb

def log_metrics(metrics: dict, 
                step: int,
                args,
                wandb = None, 
                train = True,
                ) -> None:
    metrics['Accuracy'] = metrics['Accuracy'] / metrics['total']

    if train:
        print(step, "Loss:", round(metrics['Total Loss'].item(), 2))

    print('In Logging', wandb, args.rank)
    if wandb and args.rank == 0:
        if not train:
            new_metrics = {}
            for i in metrics:
                if 'Accuracy' in i:
                    new_metrics['Val ' + i] = metrics[i]
            print('logging')
            wandb.log(new_metrics, step = step)
        else:
            wandb.log(metrics, step = step)
        
    return