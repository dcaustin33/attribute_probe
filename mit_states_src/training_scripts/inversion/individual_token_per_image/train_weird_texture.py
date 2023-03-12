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
import wandb

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
#from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='red-ball-pics-4')
parser.add_argument('--output_dir', type=str, default='output/red-ball-pics-4')
parser.add_argument('-train', action='store_true')
parser.add_argument('-resized_crop', action='store_true')
parser.add_argument('--name', type=str)
parser.add_argument('--steps', type=int, default=3000)
args = parser.parse_args()
if args.output_dir[-1] != '/':
  args.output_dir += '/'

pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        im = img.resize((w//4, h//4))
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# %%
from torchvision.datasets.folder import default_loader
directory = args.data_dir


def convert_rgb(img):
  return default_loader(img)

images = []
for i in os.listdir(directory):
  images.append(convert_rgb(directory + '/' + i))

grid = image_grid(images, 1, len(images))
if os.path.exists(args.output_dir) == False:
    os.mkdir(args.output_dir)
grid.save(args.output_dir + 'training_images.png')


# %%
#@title Settings for your newly created concept
#@markdown `what_to_teach`: what is it that you are teaching? `object` enables you to teach the model a new object to be used, `style` allows you to teach the model a new style one can use.
what_to_teach = "object" #@param ["object", "style"]
#@markdown `placeholder_token` is the token you are going to use to represent your new concept (so when you prompt the model, you will say "A `<my-placeholder-token>` in an amusement park"). We use angle brackets to differentiate a token from other words/tokens, to avoid collision.
placeholder_token1 = "<color>" #@param {type:"string"}
image_tokens = []
for i in range(len(images)):
    image_tokens.append('<image' + str(i) + ">")

#@markdown `initializer_token` is a word that can summarise what your new concept is, to be used as a starting point
initializer_token1 = "object" #@param {type:"string"}
initializer_token2 = "color" #@param {type:"string"}

#@title Setup the prompt templates for training 
#@title Setup the prompt templates for training 

imagenet_templates_small = [
"a photo of a small {} with the main concept being {}",
"a rendering of a small {} with the main concept being {}",
"a cropped photo of a small {} with the main concept being {}",
"the photo of a small {} with the main concept being {}",
"a photo of a clean small {} with the main concept being {}",
"a photo of a dirty small {} with the main concept being {}",
"a dark photo of a small {} with the main concept being {}",
"a photo of a small {} with the main concept being {}",
"a close-up photo of a small {} with the main concept being {}",
"a bright photo of a small {} with the main concept being {}",
"a good photo of a small {} with the main concept being {}",
"a photo of one small {} with the main concept being {}",
"a close-up photo of a small {} with the main concept being {}",
"a rendition of a small {} with the main concept being {}",
"a photo of a nice small {} with the main concept being {}",
"a good photo of a small {} with the main concept being {}",
"a photo of a weird small {} with the main concept being {}",
"a photo of a large small {} with the main concept being {}",
"a photo of a cool small {} with the main concept being {}",
"a photo of a small small {} with the main concept being {}"
]


#@title Setup the dataset
class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token1="*",
        center_crop=False,
    ):

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token1 = placeholder_token1
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")
        image_index = i % self.num_images

        image_token = image_tokens[image_index]

        placeholder_string1 = self.placeholder_token1
        text = random.choice(self.templates).format(image_token, placeholder_string1)
        example['text'] = text

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        #small change

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
        

        image = Image.fromarray(img)
        #random resize crop
        if args.resized_crop:
            image = transforms.RandomResizedCrop(size = self.size, scale = (.25, .75))(image)

        image = image.resize((self.size, self.size), resample=self.interpolation)


        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example

# %%
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
    
token_ids1 = tokenizer.encode(initializer_token1, add_special_tokens=False)
# Check if initializer_token is a single token or a sequence of tokens
if len(token_ids1) > 1:
    raise ValueError("The initializer token must be a single token.")

initializer_token_id1 = token_ids1[0]
placeholder_token_id1 = tokenizer.convert_tokens_to_ids(placeholder_token1)

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


for token in image_tokens:
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens < 1:
        raise ValueError(
            f"The tokenizer already contains the token {token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

text_encoder.resize_token_embeddings(len(tokenizer))
token_embeds = text_encoder.get_input_embeddings().weight.data
token_embeds[placeholder_token_id1] = token_embeds[initializer_token_id1]

for token in image_tokens:
    token_ids = tokenizer.encode(initializer_token2, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")
    
    initializer_token_id2 = token_ids[0]
    placeholder_token_id2 = tokenizer.convert_tokens_to_ids(token)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds[placeholder_token_id2] = token_embeds[initializer_token_id2]



# %%
def freeze_params(params):
    for param in params:
        param.requires_grad = False

# Freeze vae and unet
freeze_params(vae.parameters())
freeze_params(unet.parameters())
# Freeze all parameters except for the token embeddings in text encoder
params_to_freeze = itertools.chain(
    text_encoder.text_model.encoder.parameters(),
    text_encoder.text_model.final_layer_norm.parameters(),
    text_encoder.text_model.embeddings.position_embedding.parameters(),
)
freeze_params(params_to_freeze)

# %%
train_dataset = TextualInversionDataset(
      data_root=directory,
      tokenizer=tokenizer,
      size=512,
      placeholder_token1=placeholder_token1,
      repeats=100,
      learnable_property=what_to_teach, #Option selected above between object and style
      center_crop=False,
      set="train",
)
def create_dataloader(train_batch_size=1):
    return torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

noise_scheduler = DDPMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)

hyperparameters = {
    "learning_rate": 5e-04,
    "scale_lr": True,
    "max_train_steps": args.steps,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "seed": 42,
    "output_dir": args.name,
}
if args.train:
    wandb = wandb.init(config = args, name = args.name, project = 'attribute-inversion')
# %%
def training_function(text_encoder, vae, unet, wandb):
    logger = get_logger(__name__)
    
    train_batch_size = hyperparameters["train_batch_size"]
    gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
    learning_rate = hyperparameters["learning_rate"]
    max_train_steps = hyperparameters["max_train_steps"]
    output_dir = hyperparameters["output_dir"]

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    train_dataloader = create_dataloader(train_batch_size)

    if hyperparameters["scale_lr"]:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=learning_rate,
    )


    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader
    )

    # Move vae and unet to device
    vae.to(accelerator.device)
    unet.to(accelerator.device)

    # Keep vae and unet in eval model as we don't train these
    vae.eval()
    unet.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    logging_loss = 0
    logging_steps = 0

    for epoch in range(num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                logging_loss += loss.item()
                logging_steps += 1
                accelerator.backward(loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if accelerator.num_processes > 1:
                    grads = text_encoder.module.get_input_embeddings().weight.grad
                else:
                    grads = text_encoder.get_input_embeddings().weight.grad
                # Get the index for tokens that we want to zero the grads for
                index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id1 #& torch.arange(len(tokenizer)) != placeholder_token_id2
                index_grads_to_zero[placeholder_token_id1] = False
                grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                optimizer.step()
                optimizer.zero_grad()

                if logging_steps % 100 == 0:
                    wandb.log({"loss": logging_loss / logging_steps})
                    logging_loss = 0
                    logging_steps = 0

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

            if step % 1000 == 0:
                pipeline = StableDiffusionPipeline(
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    vae=vae,
                    unet=unet,
                    tokenizer=tokenizer,
                    scheduler=PNDMScheduler(
                        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
                    ),
                    safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
                    feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
                )
                pipeline.save_pretrained(output_dir + '/step_' + str(step))
                # Also save the newly trained embeddings
                learned_embeds1 = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id1]
                learned_embeds_dict = {placeholder_token1: learned_embeds1.detach().cpu()}
                torch.save(learned_embeds_dict, os.path.join(output_dir + '/step_' + str(step), "learned_embeds.bin"))
                


        accelerator.wait_for_everyone()


    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline(
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=PNDMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
            ),
            safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        )
        pipeline.save_pretrained(output_dir)
        # Also save the newly trained embeddings
        learned_embeds1 = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id1]
        learned_embeds_dict = {placeholder_token1: learned_embeds1.detach().cpu()}
        torch.save(learned_embeds_dict, os.path.join(output_dir, "learned_embeds.bin"))

def visualize(pipe, prompt, num_samples, num_rows, name, step):
    all_images = [] 
    for _ in range(num_rows):
        images = pipe([prompt] * num_samples, num_inference_steps=1000, temperature=0.99, top_k=0, top_p=0.9)
        all_images.extend(images)
    grid = image_grid(all_images, num_samples, num_rows)
    grid.save(args.output_dir + name + "_step_{}.png".format(step))

import accelerate
if args.train:
    accelerate.notebook_launcher(training_function, args=(text_encoder, vae, unet, wandb), num_processes = 1)



if os.path.exists(args.output_dir) == False:
    os.mkdir(args.output_dir)

#@title Set up the pipeline 
pipe = StableDiffusionPipeline.from_pretrained(
    hyperparameters["output_dir"],
    torch_dtype=torch.float16,
).to("cuda")

prompt = "a photo of a person with {}".format(placeholder_token1) #@param {type:"string"}

num_samples = 4 #@param {type:"number"}
num_rows = 1 #@param {type:"number"}

all_images = [] 
for _ in range(num_rows):
    images = pipe([prompt] * num_samples, num_inference_steps=50, guidance_scale=7.5).images
    all_images.extend(images)

grid = image_grid(all_images, num_samples, num_rows)
grid.save(args.output_dir + "person.png")

prompt = "a photo of a bird with {}".format(placeholder_token1) #@param {type:"string"}

num_samples = 4 #@param {type:"number"}
num_rows = 1 #@param {type:"number"}

all_images = [] 
for _ in range(num_rows):
    images = pipe([prompt] * num_samples, num_inference_steps=50, guidance_scale=7.5).images
    all_images.extend(images)

grid = image_grid(all_images, num_samples, num_rows)
grid.save(args.output_dir + "bird.png")

prompt = "a photo of a cup with {}".format(placeholder_token1) #@param {type:"string"}

num_samples = 4 #@param {type:"number"}
num_rows = 1 #@param {type:"number"}

all_images = [] 
for _ in range(num_rows):
    images = pipe([prompt] * num_samples, num_inference_steps=50, guidance_scale=7.5).images
    all_images.extend(images)

grid = image_grid(all_images, num_samples, num_rows)
grid.save(args.output_dir + "cup.png")

prompt = "a photo of a person's hair with the texture {}".format(placeholder_token1) #@param {type:"string"}

num_samples = 4 #@param {type:"number"}
num_rows = 1 #@param {type:"number"}

all_images = [] 
for _ in range(num_rows):
    images = pipe([prompt] * num_samples, num_inference_steps=50, guidance_scale=7.5).images
    all_images.extend(images)

grid = image_grid(all_images, num_samples, num_rows)
grid.save(args.output_dir + "hair.png")


prompt = "a photo of a drapes with the texture {}".format(placeholder_token1) #@param {type:"string"}

num_samples = 4 #@param {type:"number"}
num_rows = 1 #@param {type:"number"}

all_images = [] 
for _ in range(num_rows):
    images = pipe([prompt] * num_samples, num_inference_steps=50, guidance_scale=7.5).images
    all_images.extend(images)

grid = image_grid(all_images, num_samples, num_rows)
grid.save(args.output_dir + "drapes.png")

