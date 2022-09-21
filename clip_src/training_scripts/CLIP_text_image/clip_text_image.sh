export PYTHONPATH="../../";
export OMP_NUM_THREADS=4;
torchrun --nproc_per_node 1 train_clip_text_image.py \
                            --batch_size 128 \
                            --dist_url env:// \
                            --name CLIP_text_image \
                            --workers 6 \
                            --log_n_train_steps 500 \
                            --log_n_steps 500 \
                            --lr 0.001 \
                            --steps 4000 \
                            -log;

torchrun --nproc_per_node 1 eval_clip_text_image.py \
                            --batch_size 256 \
                            --name Eval_CLIP_text_image \
                            --dist_url env:// \
                            --workers 8 \
                            --saved_path checkpoints/CLIP_text_image/CLIP_text_image_Final.pt \
                            -log;

#sudo shutdown -h;