export PYTHONPATH="../../";
export OMP_NUM_THREADS=4;
torchrun --nproc_per_node 1 train_byol.py \
                            --batch_size 128 \
                            --dist_url env:// \
                            --name BYOL_Linear \
                            --workers 6 \
                            --log_n_train_steps 300 \
                            --log_n_steps 300 \
                            --lr 0.001 \
                            --steps 4000 \
                            --certainty_threshold 3 \
                            -log;

torchrun --nproc_per_node 1 eval_byol.py \
                            --batch_size 256 \
                            --name Eval_BYOL_Linear \
                            --dist_url env:// \
                            --workers 8 \
                            --certainty_threshold 3 \
                            --saved_path checkpoints/BYOL_Linear/BYOL_Linear_Final.pt \
                            -log;

torchrun --nproc_per_node 1 train_byol.py \
                            --batch_size 128 \
                            --dist_url env:// \
                            --name BYOL_Linear_2 \
                            --workers 6 \
                            --log_n_train_steps 300 \
                            --log_n_steps 300 \
                            --lr 0.001 \
                            --steps 4000 \
                            --certainty_threshold 2 \
                            -log;

torchrun --nproc_per_node 1 eval_byol.py \
                            --batch_size 256 \
                            --name Eval_BYOL_Linear_2\
                            --dist_url env:// \
                            --workers 8 \
                            --certainty_threshold 2 \
                            --saved_path checkpoints/BYOL_Linear_2/BYOL_Linear_2_Final.pt \
                            -log;

#sudo shutdown -h;