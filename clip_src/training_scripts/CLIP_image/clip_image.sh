export PYTHONPATH="../../";
export OMP_NUM_THREADS=4;
torchrun --nproc_per_node 1 train_clip_image.py \
                            --batch_size 256 \
                            --dist_url env:// \
                            --name CUB_CLIP_image_linear \
                            --workers 6 \
                            --log_n_train_steps 300 \
                            --log_n_steps 300 \
                            --lr 0.001 \
                            --certainty_threshold 3 \
                            --steps 4000 \
                            -log;

torchrun --nproc_per_node 1 eval_clip_image.py \
                            --batch_size 256 \
                            --name Eval_CUB_CLIP_image_linear\
                            --dist_url env:// \
                            --workers 8 \
                            --certainty_threshold 3 \
                            --saved_path checkpoints/CUB_CLIP_image_linear/CUB_CLIP_image_linear_Final.pt \
                            -log;

#sudo shutdown -h;