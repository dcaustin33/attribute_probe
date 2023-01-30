export PYTHONPATH="../../";
export OMP_NUM_THREADS=4;
torchrun --rdzv_endpoint=localhost:2015 --nproc_per_node 1 clip_zero_shot.py \
                            --batch_size 256 \
                            --dist_url env:// \
                            --name MLP_CUB \
                            --workers 6;