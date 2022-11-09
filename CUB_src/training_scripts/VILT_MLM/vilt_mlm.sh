export PYTHONPATH="../../";
export OMP_NUM_THREADS=4;

#torchrun --nproc_per_node 1 train_vilt_mlm.py \
#                            --batch_size 64 \
#                            --dist_url env:// \
#                            --name CUB_ViLT_mlm \
#                            --workers 6 \
#                            --log_n_train_steps 300 \
#                            --log_n_steps 100 \
#                            --lr 0.001 \
#                            --steps 4000 \
#                            --certainty_threshold 3 \
#                            -log;

torchrun --nproc_per_node 1 eval_vilt_mlm.py \
                           --batch_size 256 \
                           --name Eval_CUB_ViLT_off_shelf \
                           --dist_url env:// \
                           --workers 8 \
                           --certainty_threshold 2 \
                           --saved_path checkpoints/CUB_ViLT_mlm_off_shelf/CUB_ViLT_mlm_off_shelf_Final.pt \
                           -log;