### 1st step: get the background mask and inversion files for DPL
IMG_FORMAT='jpg'
IMG_FOLDER='images/'
FILE_NAME='catdog'

# CUDA_VISIBLE_DEVICES=1 python _2_DDIM_inv.py \
#     --input_image ${IMG_FOLDER}/${FILE_NAME}.${IMG_FORMAT} \
#     --results_folder output/ \

### 2nd step: Dynamic Prompt Learning
PLACEHOLDER1='<cat-toy>'
PLACEHOLDER2='<dog-toy>'

INIT_TOKEN1='cat'
INIT_TOKEN2='dog'

MAX_ITER=0

lam_cos=4.0 
lam_iou=4.0
lam_kl=4.0

CUDA_VISIBLE_DEVICES=0 python _3_dpl_seg_inv.py \
    --input_image ${IMG_FOLDER}/${FILE_NAME}.${IMG_FORMAT} \
    --results_folder output/${FILE_NAME}/ \
    --max_iter_to_alter ${MAX_ITER} \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2}   \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2}   \
    --lam_cos  ${lam_cos} \
    --lam_iou  ${lam_iou} \
    --lam_kl  ${lam_kl} \
    --smooth_op \
    --softmax_op \
    --seg_dirs seg_dirs/${FILE_NAME} \
