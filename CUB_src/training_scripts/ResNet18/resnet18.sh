export PYTHONPATH="../../";
export OMP_NUM_THREADS=4;
torchrun --nproc_per_node 1 train_resnet18.py \
                            --batch_size 128 \
                            --dist_url env:// \
                            --name ResNet18 \
                            --workers 6 \
                            --log_n_train_steps 200 \
                            --log_n_steps 200 \
                            --lr 0.001 \
                            --steps 8000 \
                            -log;

torchrun --nproc_per_node 1 eval_resnet18.py \
                            --batch_size 256 \
                            --name Eval_ResNet18 \
                            --dist_url env:// \
                            --workers 8 \
                            --saved_path checkpoints/ResNet18/ResNet18_Final.pt \
                            -log;

sudo shutdown -h;