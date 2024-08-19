import torch
from diffusers import DDIMScheduler
from PIL import Image
from pipelines.seg_null_textinv_pipeline import StableDiffusion_SegPipeline
from _utils.ptp_utils import show_cross_attention, show_cross_attention_plus_orig_img,show_cross_attention_blackwhite, save_attn_avg, mean_iou

import argparse
import os
import pickle as pkl
import numpy as np
import copy
  
def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--e1_path', type=str, default=None)
    parser.add_argument('--e2_path', type=str, default=None)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--number', type=int, default=4)
    parser.add_argument('--prompt', type=str)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = arguments()
    torch_dtype = torch.float32
    sd_model_ckpt = args.model_path

    pipeline = StableDiffusion_SegPipeline.from_pretrained(
        sd_model_ckpt,
        torch_dtype=torch_dtype,
    )

    
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    _ = tokenizer.add_tokens(['<e1>', '<e2>'])
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_ids = tokenizer.convert_tokens_to_ids(['<e1>', '<e2>'])   
    assert len(token_ids) == 2, len(token_ids)
    token_embeds = text_encoder.get_input_embeddings().weight.data

    if args.e1_path is not None:
        e1_file = open(args.e1_path, 'rb')
        e1_emb = pkl.load(e1_file)[-1][0]
        token_embeds[token_ids[0]] = e1_emb
    if args.e2_path is not None:
        e2_file = open(args.e2_path, 'rb')
        e2_emb = pkl.load(e2_file)[-1][0]
        token_embeds[token_ids[1]] = e2_emb
    
    prompt = args.prompt
    pipeline.to("cuda")

    for x in range(args.number):

        # Generate the image
        image = pipeline(prompt).images[0]
        directory = os.path.dirname(args.save_name)

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the image
        image.save(f'{args.save_name[:-4]}-{x}.png')
