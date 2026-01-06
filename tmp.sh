# Create the results folder
mkdir -p results/dimension_scaling_d10

# 1. ZOD-MC (full rejection — will explode after d=5)
python experiments.py --mode dimension --dimension 20 --num_batches 20 --sampling_batch_size 10 --density gmm --save_folder results/dimension_scaling_d10/zod_mc --methods_to_run ZOD_MC --sampling_method ei --disc_steps 100 --T 2.0 --num_estimator_batches 8 --num_estimator_samples 100 --sampling_eps 5e-3

# 2. RDMC (ULA-in-posterior — much better than ZOD-MC)
# python experiments.py --mode dimension --dimension 10 --num_batches 20 --sampling_batch_size 10 --density gmm --save_folder results/dimension_scaling_d10/rdmc --methods_to_run RDMC --sampling_method ei --disc_steps 100 --T 2.0 --ula_step_size 0.1 --num_sampler_iterations 10 --sampling_eps 5e-2

# # 3. P_ZOD_MC (Partial ZOD-MC — the hero, stays flat!)
# python experiments.py --mode dimension --dimension 10 --num_batches 20 --sampling_batch_size 10 --density gmm --save_folder results/dimension_scaling_d10/pzod_mc --methods_to_run P_ZOD_MC --sampling_method ei --disc_steps 100 --T 2.0 --num_estimator_batches 8 --num_estimator_samples 100 --sampling_eps 5e-3

# # 4. SLIPS (state-of-the-art Langevin-based — strong baseline)
# python experiments.py --mode dimension --dimension 10 --num_batches 20 --sampling_batch_size 10 --density gmm --save_folder results/dimension_scaling_d10/SLIPS --methods_to_run SLIPS --sampling_method ei --disc_steps 100 --T 2.0

# Done! Plot is automatically generated
echo "All done! Open: results/dimension_scaling_d10/dimension_mmd_results.pdf"