export PYTHONPATH="../../";
export OMP_NUM_THREADS=4;
torchrun --nproc_per_node 1 train_mlp_with_attributes.py \
                            --batch_size 256 \
                            --dist_url env:// \
                            --name MLP_CUB \
                            --workers 6 \
                            --log_n_train_steps 10 \
                            --log_n_steps 10 \
                            --lr 0.001 \
                            --steps 4000 \
                            -log;