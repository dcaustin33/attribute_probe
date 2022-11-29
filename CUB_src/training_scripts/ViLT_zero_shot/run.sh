export PYTHONPATH="../../";
export OMP_NUM_THREADS=4;
torchrun --nproc_per_node 1 vilt_zero_shot.py \
                            --batch_size 256 \
                            --dist_url env:// \
                            --name MLP_CUB \
                            --workers 6;