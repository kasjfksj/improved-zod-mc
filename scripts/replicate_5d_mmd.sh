python3 experiments.py --config config/mmd_experiments_5d.yaml \
    --mode eval_mmd --T 2 --disc_steps 50 \
    --sampling_method ei \
    --sampling_batch_size 100 --num_batches 1 \
    --score_method p0t --density gmm --dimension 15 \
    --density_parameters_path config/density_parameters/5d_gmm.yaml \
    --save_folder plots/gmm_5d/