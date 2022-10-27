export PYTHONPATH="../../";
export OMP_NUM_THREADS=4;
torchrun --nproc_per_node 1 train_vilt_with_attribute.py \
                            --batch_size 64 \
                            --dist_url env:// \
                            --name CUB_ViLT_with_3_attributes \
                            --workers 6 \
                            --log_n_train_steps 300 \
                            --log_n_steps 300 \
                            --lr 0.001 \
                            --steps 4000 \
                            --attribute_idx_amount 3 \
                            --certainty_threshold 3 \
                            -log;

torchrun --nproc_per_node 1 eval_with_attribute.py \
                           --batch_size 256 \
                           --name CUB_ViLT_with_3_attributes \
                           --dist_url env:// \
                           --workers 8 \
                           --certainty_threshold 2 \
                           --saved_path checkpoints/CUB_ViLT_with_3_attributes/CUB_ViLT_with_3_attributes_Final.pt \
                           -log;
torchrun --nproc_per_node 1 train_vilt_with_attribute.py \
                            --batch_size 64 \
                            --dist_url env:// \
                            --name CUB_ViLT_with_0_attributes \
                            --workers 6 \
                            --log_n_train_steps 300 \
                            --log_n_steps 300 \
                            --lr 0.001 \
                            --steps 4000 \
                            --attribute_idx_amount 0 \
                            --certainty_threshold 3 \
                            -log;

torchrun --nproc_per_node 1 eval_with_attribute.py \
                           --batch_size 256 \
                           --name CUB_ViLT_with_0_attributes \
                           --dist_url env:// \
                           --workers 8 \
                           --certainty_threshold 2 \
                           --saved_path checkpoints/CUB_ViLT_with_0_attributes/CUB_ViLT_with_0_attributes_Final.pt \
                           -log;



sudo shutdown -h;