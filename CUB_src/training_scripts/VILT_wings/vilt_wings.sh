export PYTHONPATH="../../";
export OMP_NUM_THREADS=4;

torchrun --nproc_per_node 1 train_vilt_wings.py \
                           --batch_size 64 \
                           --dist_url env:// \
                           --name CUB_ViLT_wings_linear_diff_prompt \
                           --workers 6 \
                           --log_n_train_steps 2000 \
                           --log_n_steps 2000 \
                           --lr 0.001 \
                           --steps 4000 \
                           --certainty_threshold 3 \
                           -log;

torchrun --nproc_per_node 1 eval_vilt_wings.py \
                           --batch_size 256 \
                           --name Eval_CUB_ViLT_wings_linear_diff_prompt \
                           --dist_url env:// \
                           --workers 8 \
                           --certainty_threshold 2 \
                           --saved_path checkpoints/CUB_ViLT_wings_linear_diff_prompt/CUB_ViLT_wings_linear_diff_prompt_Final.pt \
                           -log;