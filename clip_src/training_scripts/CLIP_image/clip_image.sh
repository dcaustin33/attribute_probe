export PYTHONPATH="../../";
export OMP_NUM_THREADS=4;
torchrun --nproc_per_node 1 train_clip_image.py \
                            --batch_size 128 \
                            --dist_url env:// \
                            --name CLIP_image \
                            --workers 6 \
                            --log_n_train_steps 500 \
                            --log_n_steps 500 \
                            --lr 0.001 \
                            --steps 8000 \
                            -log;

sudo shutdown -h;