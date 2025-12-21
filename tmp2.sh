# Create the results folder
mkdir -p results/eval_mmd_scaling_d10

# 1. ZOD-MC (full rejection — will explode after d=5)
python experiments.py --mode eval_mmd --num_batches 20 --sampling_batch_size 10 --density gmm --dimension 20 --save_folder results/eval_mmd_scaling_d10/zod_mc --methods_to_run ZOD_MC --sampling_method ei --disc_steps 1000 --T 2.0 --num_estimator_batches 8 --num_estimator_samples 100 --sampling_eps 5e-3 --density_parameters_path config/density_parameters/5d_gmm.yaml --num_samples_for_rdmc 200 --min_num_iters_rdmc 5 --max_num_iters_rdmc 50


# Done! Plot is automatically generated
echo "All done! Open: results/eval_mmd_scaling_d10/eval_mmd_mmd_results.pdf"