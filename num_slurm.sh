#!/bin/bash

# GPU_INFO="--partition viscam --account viscam --gpu_type 3090 --cpus_per_task 8 --num_gpus 1 --mem 100G"
GPU_INFO="--partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 24G"
# GPU_INFO="--partition viscam --account viscam --gpu_type titanrtx --cpus_per_task 8 --num_gpus 1 --mem 64G"

# GPU_INFO="--partition svl --account viscam --gpu_type titanrtx --cpus_per_task 8 --num_gpus 1 --mem 64G"

# EXTRA_GPU_INFO="exclude=viscam1,viscam5,viscam7,svl[1-6],svl[8-10]"


python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
--proj_dir /viscam/projects/image2Blender/differentiable_engine/LocInv --conda_env dpl \
--job "07-28-locinv-chair" --command "python _2_DDIM_inv.py --input_image images/rendered_chair.jpg --results_folder ./output --num_ddim_steps 1 && python _3_dpl_seg_inv.py --input_image images/rendered_chair.jpg --results_folder output/rendered_chair --max_iter_to_alter 0 --initializer_token chair --placeholder_token  '<chair>'  --lam_cos 1.0 --lam_iou 1.0 --lam_kl 1.0 --lam_adj 0.0 --smooth_op --softmax_op --seg_dirs seg_dirs/rendered_chair --target_image images/rendered_chair.jpg --num_ddim_steps 1 --exp_name one_word" $GPU_INFO

python _3_dpl_seg_inv.py --input_image images/rendered_chair.jpg --results_folder output/rendered_chair --max_iter_to_alter 0 --initializer_token chair --placeholder_token  '<chair>'  --lam_cos 1.0 --lam_iou 1.0 --lam_kl 1.0 --lam_adj 0.0 --smooth_op --softmax_op --seg_dirs seg_dirs/rendered_chair --target_image images/rendered_chair.jpg --num_ddim_steps 1 --exp_name one_word && python _3_dpl_seg_inv.py --input_image images/rendered_chair.jpg --results_folder output/rendered_chair --max_iter_to_alter 0 --initializer_token one two three four five six --placeholder_token  '<one>' '<two>' '<three>' '<four>' '<five>' '<six>'  --lam_cos 1.0 --lam_iou 1.0 --lam_kl 1.0 --lam_adj 0.0 --smooth_op --softmax_op --seg_dirs seg_dirs/rendered_chair --target_image images/rendered_chair.jpg --num_ddim_steps 1 --prompt_file chair_numbers.txt --exp_name six_words --attn_inner_steps 100