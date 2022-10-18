export PYTHONPATH="../../";
export OMP_NUM_THREADS=4;
#torchrun --nproc_per_node 1 train_clip_with_attribute.py \
#                            --batch_size 256 \
#                            --dist_url env:// \
#                            --name CLIP_with_attribute \
#                            --workers 6 \
#                            --log_n_train_steps 30 \
#                            --log_n_steps 30 \
#                            --lr 0.001 \
#                            --steps 40 \
#                            --attribute_idx_amount 2 \
#                            --certainty_threshold 2;# \
#                            #-log;

torchrun --nproc_per_node 1 eval_with_attribute.py \
                            --batch_size 256 \
                            --name Eval_CLIP_with_attribute \
                            --dist_url env:// \
                            --workers 8 \
                            --certainty_threshold 2 \
                            --saved_path checkpoints/CLIP_with_attribute/CLIP_with_attribute_checkpoint.pt;# \
                            #-log;