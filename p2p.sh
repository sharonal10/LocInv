### 1st step: get the background mask and inversion files for DPL
IMG_FORMAT='jpg'
IMG_FOLDER='images/'
FILE_NAME='rendered_chair'

# CUDA_VISIBLE_DEVICES=1 python _2_DDIM_inv.py \
#     --input_image ${IMG_FOLDER}/${FILE_NAME}.${IMG_FORMAT} \
#     --results_folder output/ \

### 2nd step: Dynamic Prompt Learning
PLACEHOLDER1='<chair>'
# PLACEHOLDER2='<dog-toy>'

INIT_TOKEN1='chair'
# INIT_TOKEN2='dog'

MAX_ITER=0

lam_cos=5.0 
lam_iou=5.0
lam_kl=5.0

# CUDA_VISIBLE_DEVICES=0 python _3_dpl_seg_inv.py \
#     --input_image ${IMG_FOLDER}/${FILE_NAME}.${IMG_FORMAT} \
#     --results_folder output/${FILE_NAME}/ \
#     --max_iter_to_alter ${MAX_ITER} \
#     --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2}   \
#     --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2}   \
#     --lam_cos  ${lam_cos} \
#     --lam_iou  ${lam_iou} \
#     --lam_kl  ${lam_kl} \
#     --smooth_op \
#     --softmax_op \
#     --seg_dirs seg_dirs/${FILE_NAME} \


python _4_image_edit.py \
--input_image ${IMG_FOLDER}/${FILE_NAME}.${IMG_FORMAT} \
--postfix cos_al_50.0_beta_0.7_lam_5.0_iou_al_25.0_beta_0.7_lam_5.0_kl_al_25.0_beta_1.0_lam_5.0_adj_al_50.0_beta_0.1_lam_2.0_softmax_True_smooth_True_null_31_attn_31_CFG_7.5_adj_False \
--results_folder output/${FILE_NAME}/ \
--results_folder_edit output/test \
--negative_guidance_scale 7.5 \
--placeholder_token ${PLACEHOLDER1} \
--initializer_token ${INIT_TOKEN1} \
--replaced_embed_folder output/rendered_chair