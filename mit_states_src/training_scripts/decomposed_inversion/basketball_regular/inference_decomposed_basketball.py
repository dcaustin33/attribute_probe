# %%
#@title Import required libraries
import argparse
import itertools
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"


#@title Settings for your newly created concept
#@markdown `what_to_teach`: what is it that you are teaching? `object` enables you to teach the model a new object to be used, `style` allows you to teach the model a new style one can use.
what_to_teach = "object" #@param ["object", "style"]
#@markdown `placeholder_token` is the token you are going to use to represent your new concept (so when you prompt the model, you will say "A `<my-placeholder-token>` in an amusement park"). We use angle brackets to differentiate a token from other words/tokens, to avoid collision.
placeholder_token1 = "<shape>" #@param {type:"string"}
placeholder_token2 = "<color>" #@param {type:"string"}
#@markdown `initializer_token` is a word that can summarise what your new concept is, to be used as a starting point
initializer_token1 = "shape" #@param {type:"string"}
initializer_token2 = "color" #@param {type:"string"}

#@title Setup the prompt templates for training 
#@title Setup the prompt templates for training 
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of a {}",
    "the photo of a {}",
    "a photo of clean a {}",
    "a photo of dirty a {}",
    "a dark photo of a {}",
    "a photo of a {}",
    "a close-up photo of a {}",
    "a bright photo of a {}",
    "a good photo of a {}",
    "a photo of one a {}",
    "a close-up photo of a {}",
    "a rendition of a {}",
    "a photo of nice a {}",
    "a good photo of a {}",
    "a photo of a weird {}",
    "a photo of a large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

#@title Load the tokenizer and add the placeholder token as a additional special token.
#@markdown Please read and if you agree accept the LICENSE [here](https://huggingface.co/CompVis/stable-diffusion-v1-4) if you see an error
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
)

# Add the placeholder token in tokenizer
num_added_tokens = tokenizer.add_tokens(placeholder_token1)
if num_added_tokens < 1:
    raise ValueError(
        f"The tokenizer already contains the token {placeholder_token1}. Please pass a different"
        " `placeholder_token` that is not already in the tokenizer."
    )
num_added_tokens = tokenizer.add_tokens(placeholder_token2)
if num_added_tokens < 1:
    raise ValueError(
        f"The tokenizer already contains the token {placeholder_token2}. Please pass a different"
        " `placeholder_token` that is not already in the tokenizer."
    )
    
token_ids1 = tokenizer.encode(initializer_token1, add_special_tokens=False)
token_ids2 = tokenizer.encode(initializer_token2, add_special_tokens=False)
# Check if initializer_token is a single token or a sequence of tokens
if len(token_ids1) > 1:
    raise ValueError("The initializer token must be a single token.")
if len(token_ids2) > 1:
    raise ValueError("The initializer token must be a single token.")

initializer_token_id1 = token_ids1[0]
placeholder_token_id1 = tokenizer.convert_tokens_to_ids(placeholder_token1)
initializer_token_id2 = token_ids2[0]
placeholder_token_id2 = tokenizer.convert_tokens_to_ids(placeholder_token2)


#@title Load the Stable Diffusion model
# Load models and create wrapper for stable diffusion
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae"
)
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet"
)
text_encoder.resize_token_embeddings(len(tokenizer))
token_embeds = text_encoder.get_input_embeddings().weight.data
token_embeds[placeholder_token_id1] = token_embeds[initializer_token_id1]
token_embeds[placeholder_token_id2] = token_embeds[initializer_token_id2]


noise_scheduler = DDPMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, tensor_format="pt"
)

hyperparameters = {
    "learning_rate": 5e-04,
    "scale_lr": True,
    "max_train_steps": 3000,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "seed": 42,
    "output_dir": "decomposed-basketball-concept"
}


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        im = img.resize((w//4, h//4))
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


#@title Set up the pipeline 
pipe = StableDiffusionPipeline.from_pretrained(
    hyperparameters["output_dir"],
    torch_dtype=torch.float16,
).to("cuda")

prompt = "a picture of a {} with the color {}".format(placeholder_token1, placeholder_token2) #@param {type:"string"}

num_samples = 4 #@param {type:"number"}
num_rows = 1 #@param {type:"number"}

all_images = [] 
for _ in range(num_rows):
    images = pipe([prompt] * num_samples, num_inference_steps=100, guidance_scale=7.5).images
    all_images.extend(images)

grid = image_grid(all_images, num_samples, num_rows)
#save grid as an image
grid.save("output_decomposed_grid.png")


print('Cosine Distance')
print()
import torch.nn.functional as F
import torch.nn as nn
embeddings = pipe.text_encoder.get_input_embeddings().weight
objects_to_compare_to = ['volleyball', 'basketball', 'baseball', 'football', 'soccer', 'tennis']
token_id = pipe.tokenizer.convert_tokens_to_ids(placeholder_token1)
compare_embeddings = pipe.text_encoder.get_input_embeddings().weight[pipe.tokenizer.convert_tokens_to_ids(objects_to_compare_to)]
cos = nn.CosineSimilarity(dim=1, eps=1e-3)
output = cos(embeddings[token_id].unsqueeze(dim = 0), compare_embeddings)
val, idx = torch.topk(output, k = len(objects_to_compare_to))
print(val)
print(idx)
for i, index in enumerate(idx):
    print(objects_to_compare_to[index.item()], val[i].item())

print()
print('Cosine Distance')
print()
import torch.nn.functional as F
import torch.nn as nn
embeddings = pipe.text_encoder.get_input_embeddings().weight
objects_to_compare_to = ['yellow', 'red', 'blue', 'green', 'orange', 'purple']
token_id = pipe.tokenizer.convert_tokens_to_ids(placeholder_token2)
compare_embeddings = pipe.text_encoder.get_input_embeddings().weight[pipe.tokenizer.convert_tokens_to_ids(objects_to_compare_to)]
cos = nn.CosineSimilarity(dim=1, eps=1e-3)
output = cos(embeddings[token_id].unsqueeze(dim = 0), compare_embeddings)
val, idx = torch.topk(output, k = len(objects_to_compare_to))
print(val)
print(idx)
for i, index in enumerate(idx):
    print(objects_to_compare_to[index.item()], val[i].item())




print()
print()
print()
print('Euclidean Distance')
print()

embeddings = pipe.text_encoder.get_input_embeddings().weight
objects_to_compare_to = ['volleyball', 'basketball', 'baseball', 'football', 'soccer', 'tennis']
token_id = pipe.tokenizer.convert_tokens_to_ids(placeholder_token1)
compare_embeddings = pipe.text_encoder.get_input_embeddings().weight[pipe.tokenizer.convert_tokens_to_ids(objects_to_compare_to)]
output = -torch.cdist(embeddings[token_id].unsqueeze(dim = 0).float(), compare_embeddings.float())
val, idx = torch.topk(output, k = len(objects_to_compare_to))
print(val)
print(idx)
for i, index in enumerate(idx[0]):
    print(objects_to_compare_to[index.item()], val[0][i])

print()
print('Euclidean Distance')
embeddings = pipe.text_encoder.get_input_embeddings().weight
objects_to_compare_to = ['yellow', 'red', 'blue', 'green', 'orange', 'purple']
token_id = pipe.tokenizer.convert_tokens_to_ids(placeholder_token2)
compare_embeddings = pipe.text_encoder.get_input_embeddings().weight[pipe.tokenizer.convert_tokens_to_ids(objects_to_compare_to)]
output = -torch.cdist(embeddings[token_id].unsqueeze(dim = 0).float(), compare_embeddings.float())
val, idx = torch.topk(output, k = len(objects_to_compare_to))
print(val)
print(idx)
for i, index in enumerate(idx[0]):
    print(objects_to_compare_to[index.item()], val[0][i])

