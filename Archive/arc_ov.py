# -*- coding: UTF-8 -*-

from diffusers import StableDiffusionPipeline
from pathlib import Path
import torch
import os
import numpy as np
import inspect
from typing import List, Optional, Union
from transformers import CLIPTokenizer
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from openvino.runtime import Model
from openvino.runtime import Core
from transformers import CLIPTokenizer
from diffusers.schedulers import LMSDiscreteScheduler
#import ipywidgets as widgets
import random

TEXT_ENCODER_ONNX_PATH = Path('ovTempModels/text_encoder.onnx')
TEXT_ENCODER_OV_PATH = Path('ovModels/text_encoder.xml')
UNET_ONNX_PATH = Path('ovTempModels/unet.onnx')
UNET_OV_PATH = Path('ovModels/unet.xml')
VAE_ONNX_PATH = Path('ovTempModels/vae_decoder.onnx')
VAE_OV_PATH = Path('ovModels/vae_decoder.xml')

# ### Text Encoder
# 
# The text-encoder is responsible for transforming the input prompt, e.g. "a photo of an astronaut riding a horse" into an embedding space that can be understood by the U-Net. It is usually a simple transformer-based encoder that maps a sequence of input tokens to a sequence of latent text embeddings.
# 
# Input of text encoder is tensor `input_ids` which contains indexes of tokens from text processed by tokenizer and padded to maximum length accepted by model. Model outputs are 2 tensors: `last_hidden_state` - hidden state from the last MultiHeadAttention layer in model and `pooler_out` - Pooled output for whole model hidden states. We will use `opset_version=14`, because model contains `triu` operation, supported in ONNX only starting from this opset.
def convert_encoder_onnx(pipe: StableDiffusionPipeline, onnx_path:Path):
    """
    Convert Text Encoder model to ONNX. 
    Function accepts pipeline, prepares example inputs for ONNX conversion via torch.export, 
    Parameters: 
        pipe (StableDiffusionPipeline): Stable Diffusion pipeline
        onnx_path (Path): File for storing onnx model
    Returns:
        None
    """
    if not onnx_path.exists():
        text = 'a photo of an astronaut riding a horse on mars'
        text_encoder = pipe.text_encoder
        input_ids = pipe.tokenizer(
            text,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        # switch model to inference mode
        text_encoder.eval()

        # disable gradients calculation for reducing memory consumption
        with torch.no_grad():
            # infer model, just to make sure that it works
            text_encoder(input_ids)
            # export model to ONNX format
            torch.onnx.export(
                text_encoder,  # model instance
                input_ids,  # inputs for model tracing
                onnx_path,  # output file for saving result
                input_names=['input_ids'],  # model input name for onnx representation
                output_names=['last_hidden_state', 'pooler_out'],  # model output names for onnx representation
                opset_version=14  # onnx opset version for export
            )
        print('Text Encoder successfully converted to ONNX')
    else:
        print('Text Encoder exists for ONNX')
    
def convert_encoder_ov(pipe):
    if not TEXT_ENCODER_OV_PATH.exists():
        convert_encoder_onnx(pipe, TEXT_ENCODER_ONNX_PATH)
        os.system("mo --input_model " + TEXT_ENCODER_ONNX_PATH.parent.name + "/" + TEXT_ENCODER_ONNX_PATH.name + " --output_dir " + TEXT_ENCODER_OV_PATH.parent.name + " --compress_to_fp16")
        print('Text Encoder successfully converted to IR')
    else:
        print('Text Encoder exists for IR')

# ### U-net
# 
# Unet model has 3 inputs:
# * `sample` - latent image sample from previous step, generation process has not been started yet, so we will use random noise
# * `timestep` - current scheduler step
# * `encoder_hidden_state` - hidden state of text encoder.
# 
# Model predicts the `sample` state for the next step.
def convert_unet_onnx(pipe:StableDiffusionPipeline, onnx_path:Path):
    """
    Convert Unet model to ONNX, then IR format. 
    Function accepts pipeline, prepares example inputs for ONNX conversion via torch.export, 
    Parameters: 
        pipe (StableDiffusionPipeline): Stable Diffusion pipeline
        onnx_path (Path): File for storing onnx model
    Returns:
        None
    """
    if not onnx_path.exists():
        # prepare inputs
        text = 'a photo of an astronaut riding a horse on mars'
        text_encoder = pipe.text_encoder
        input_ids = pipe.tokenizer(
            text,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        with torch.no_grad():
            text_encoder_output = text_encoder(input_ids)
        latents_shape = (2, 4, 512 // 8, 512 // 8)
        latents = torch.randn(latents_shape)
        t = torch.from_numpy(np.array(1, dtype=float))

        # model size > 2Gb, it will be represented as onnx with external data files, we will store it in separated directory for avoid a lot of files in current directory
        onnx_path.parent.mkdir(exist_ok=True, parents=True)

        max_length = input_ids.shape[-1]

        # we plan to use unet with classificator free guidence, in this cace conditionaly generated text embeddings should be concatenated with uncoditional
        uncond_input = pipe.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids)[0]
        encoder_hidden_state = torch.cat([uncond_embeddings, text_encoder_output[0]])

        # to make sure that model works
        pipe.unet(latents, t, encoder_hidden_state)[0]

        with torch.no_grad():
            torch.onnx.export(
                pipe.unet, 
                (latents, t, encoder_hidden_state), str(onnx_path),
                input_names=['sample', 'timestep', 'encoder_hidden_state'],
                output_names=['out_sample']
            )
        print('Unet successfully converted to ONNX')
    else:
        print('Unet exists for ONNX')

def convert_unet_ov(pipe):
    if not UNET_OV_PATH.exists():
        convert_unet_onnx(pipe, UNET_ONNX_PATH)
        os.system("mo --input_model " + UNET_ONNX_PATH.parent.name + "/" + UNET_ONNX_PATH.name + " --output_dir " + UNET_OV_PATH.parent.name + " --compress_to_fp16")
        print('Unet successfully converted to IR')
    else:
        print('Unet exists for IR')

# ### VAE
# 
# The VAE model has two parts, an encoder and a decoder. The encoder is used to convert the image into a low dimensional latent representation, which will serve as the input to the U-Net model. The decoder, conversely, transforms the latent representation back into an image.
# 
# During latent diffusion training, the encoder is used to get the latent representations (latents) of the images for the forward diffusion process, which applies more and more noise at each step. During inference, the denoised latents generated by the reverse diffusion process are converted back into images using the VAE decoder. As we will see during inference we **only need the VAE decoder**.
# 
# In our inference pipeline, we will need only decoding part of VAE, but forward function, which used for tracing runs encoding too. For obtaining only necessary part of model, we will wrap it into new model class.
def convert_vae_onnx(pipe:StableDiffusionPipeline, onnx_path:Path):
    """
    Convert VAE model to ONNX, then IR format. 
    Function accepts pipeline, creates wrapper class for export only necessary for inference part, 
    prepares example inputs for ONNX conversion via torch.export, 
    Parameters: 
        pipe (StableDiffusionPipeline): Stable Diffusion pipeline
        onnx_path (Path): File for storing onnx model
    Returns:
        None
    """
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae.decode(latents)

    if not onnx_path.exists():
        vae_decoder = VAEDecoderWrapper(pipe.vae)
        text = 'a photo of an astronaut riding a horse on mars'
        text_encoder = pipe.text_encoder
        input_ids = pipe.tokenizer(
            text,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        with torch.no_grad():
            text_encoder_output = text_encoder(input_ids)
        latents_shape = (2, 4, 512 // 8, 512 // 8)
        latents = torch.randn(latents_shape)
        t = torch.from_numpy(np.array(1, dtype=float))
        max_length = input_ids.shape[-1]
        uncond_input = pipe.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids)[0]
        encoder_hidden_state = torch.cat([uncond_embeddings, text_encoder_output[0]])
        output_latents = pipe.unet(latents, t, encoder_hidden_state)[0]
        latents_uncond, latents_text = output_latents[0].unsqueeze(0), output_latents[1].unsqueeze(0)
        latents = latents_uncond + 7.5 * (latents_text - latents_uncond)

        vae_decoder.eval()
        with torch.no_grad():
            torch.onnx.export(vae_decoder, latents, onnx_path, input_names=['latents'], output_names=['sample'])
        print('VAE decoder successfully converted to ONNX')
    else:
        print('VAE decoder exists for ONNX')

def convert_vae_ov(pipe):
    if not VAE_OV_PATH.exists():
        convert_vae_onnx(pipe, VAE_ONNX_PATH)
        os.system("mo --input_model " + VAE_ONNX_PATH.parent.name + "/" + VAE_ONNX_PATH.name + " --output_dir " + VAE_OV_PATH.parent.name + " --compress_to_fp16")
        print('VAE successfully converted to IR')
    else:
        print('VAE exists for IR')

# The stable diffusion model takes both a latent seed and a text prompt as an input. The latent seed is then used to generate random latent image representations of size $64 \times 64$ where as the text prompt is transformed to text embeddings of size $77 \times 768$ via CLIP's text encoder.
# 
# Next the U-Net iteratively *denoises* the random latent image representations while being conditioned on the text embeddings. The output of the U-Net, being the noise residual, is used to compute a denoised latent image representation via a scheduler algorithm. Many different scheduler algorithms can be used for this computation, each having its pros and cons. For Stable Diffusion, it is recommended using one of:
# 
# - [PNDM scheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py)
# - [DDIM scheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py)
# - [K-LMS scheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lms_discrete.py)(we will use it in our pipeline)
# 
# Theory on how the scheduler algorithm function works is out of scope for this notebook, but in short one should remember that they compute the predicted denoised image representation from the previous noise representation and the predicted noise residual.
# For more information, we recommend looking into [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
# 
# The *denoising* process is repeated given number of times (by default 50) to step-by-step retrieve better latent image representations.
# Once complete, the latent image representation is decoded by the decoder part of the variational auto encoder.
class OVStableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: Model,
        text_encoder: Model,
        tokenizer: CLIPTokenizer,
        unet: Model,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        """
        Pipeline for text-to-image generation using Stable Diffusion.
        Parameters:
            vae (Model):
                Variational Auto-Encoder (VAE) Model to decode images to and from latent representations.
            text_encoder (Model):
                Frozen text-encoder. Stable Diffusion uses the text portion of
                [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
                the clip-vit-large-patch14(https://huggingface.co/openai/clip-vit-large-patch14) variant.
            tokenizer (CLIPTokenizer):
                Tokenizer of class CLIPTokenizer(https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
            unet (Model): Conditional U-Net architecture to denoise the encoded image latents.
            scheduler (SchedulerMixin):
                A scheduler to be used in combination with unet to denoise the encoded image latents. Can be one of
                DDIMScheduler, LMSDiscreteScheduler, or PNDMScheduler.
        """
        super().__init__()
        self.scheduler = scheduler
        self.vae = vae
        self.text_encoder = text_encoder
        self.unet = unet
        self._text_encoder_output = text_encoder.output(0)
        self._unet_output = unet.output(0)
        self._vae_output = vae.output(0)
        self.height = self.unet.input(0).shape[2] * 8
        self.width = self.unet.input(0).shape[3] * 8
        self.tokenizer = tokenizer

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]],
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        latents: Optional[np.array] = None,
        output_type: Optional[str] = "pil",
        seed: Optional[int] = None,
        gif: Optional[bool] = False,
        **kwargs,
    ):
        """
        Function invoked when calling the pipeline for generation.
        Parameters:
            prompt (str or List[str]):
                The prompt or prompts to guide the image generation.
            num_inference_steps (int, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (float, *optional*, defaults to 7.5):
                Guidance scale as defined in Classifier-Free Diffusion Guidance(https://arxiv.org/abs/2207.12598).
                guidance_scale is defined as `w` of equation 2.
                Higher guidance scale encourages to generate images that are closely linked to the text prompt,
                usually at the expense of lower image quality.
            eta (float, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [DDIMScheduler], will be ignored for others.
            latents (torch.FloatTensor, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random generator.
            output_type (`str`, *optional*, defaults to "pil"):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): PIL.Image.Image or np.array.
            seed (int, *optional*, None):
                Seed for random generator state initialization.
            gif (bool, *optional*, False):
                Flag for storing all steps results or not.
        Returns:
            Dictionary with keys: 
                sample - the last generated image PIL.Image.Image or np.array
                iterations - *optional* (if gif=True) images for all diffusion steps, List of PIL.Image.Image or np.array.
        """
        if seed is not None:
            np.random.seed(seed)

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        img_buffer = []
        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_embeddings = self.text_encoder(text_input.input_ids)[self._text_encoder_output]
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if True: #Force add negative prompt #do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                negative_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="np"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[self._text_encoder_output]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it
        latents_shape = (batch_size, 4, self.height // 8, self.width // 8)
        if latents is None:
            latents = np.random.randn(
                *latents_shape
            )
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        timesteps = self.scheduler.timesteps

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0].numpy()

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet([latent_model_input, t, text_embeddings])[self._unet_output]
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs)["prev_sample"].numpy()
            if gif:
                image = self.vae(1 / 0.18215 * latents)[self._vae_output]
                image = (image / 2 + 0.5).clip(0, 1)
                image = image.transpose(0, 2, 3, 1)
                if output_type == "pil":
                    image = self.numpy_to_pil(image)
                img_buffer.extend(image)

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae(latents)[self._vae_output]

        image = (image / 2 + 0.5).clip(0, 1)
        image = image.transpose(0, 2, 3, 1)
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image, 'iterations': img_buffer}

import re
def validateTitle(title):
    rstr = r"[\/\\\:\*\?\"\<\>\|.]"
    return re.sub(rstr, r"", title)

# ### EXPORT
#
#REPO = "windwhinny/chilloutmix"
REPO = "compvis/stable-diffusion-v1-4"
#REPO = "runwayml/stable-diffusion-v1-5"

def downloadModel():
    if not TEXT_ENCODER_OV_PATH.exists() or not UNET_OV_PATH.exists() or not VAE_OV_PATH.exists():
        if not os.path.exists('ovTempModels'):
            os.mkdir('ovTempModels')
        if not os.path.exists('ovModels'):
            os.mkdir('ovModels')    
        pipe = StableDiffusionPipeline.from_pretrained(REPO, force_download=False)
        convert_encoder_ov(pipe)
        convert_unet_ov(pipe)
        convert_vae_ov(pipe)        
        del pipe
    
def compileModel(xpu):
    if xpu == 'CPU' or xpu == 'GPU':
        core = Core()
        text_enc = core.compile_model(TEXT_ENCODER_OV_PATH, xpu)
        unet_model = core.compile_model(UNET_OV_PATH, xpu)
        vae = core.compile_model(VAE_OV_PATH, xpu)
        
        '''
        lms = LMSDiscreteScheduler(
            beta_start=0.00085, 
            beta_end=0.012, 
            beta_schedule="scaled_linear"
        )
        tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
        '''
        NOINTERNET = True
        if NOINTERNET:
            lms = LMSDiscreteScheduler.from_pretrained('ovModels', subfolder="scheduler")
            tokenizer = CLIPTokenizer.from_pretrained('ovModels', subfolder="tokenizer")        
        else:
            lms = LMSDiscreteScheduler.from_pretrained(REPO, subfolder="scheduler")
            tokenizer = CLIPTokenizer.from_pretrained(REPO, subfolder="tokenizer")

        ov_pipe = OVStableDiffusionPipeline(
            tokenizer=tokenizer,
            text_encoder=text_enc,
            unet=unet_model,
            vae=vae,
            scheduler=lms
        )        
        return ov_pipe

def generateImage(xpu, ov_pipe, prompt='horse', negative='', seed=0, steps=20):
    if not os.path.exists('output'):
        os.mkdir('output')

    print('Pipeline settings')
    print(f'Input text: {prompt}')
    print(f'Negative text: {negative}')
    print(f'Seed: {seed}')
    print(f'Number of steps: {steps}')
    
    #generator = torch.manual_seed(int(seed))
    
    result = ov_pipe(prompt, negative_prompt=negative, num_inference_steps=int(steps), seed=int(seed), gif=False)

    str_image = validateTitle(str(prompt))[:40] + '_seed_' + str(seed) + '_step_' + str(steps)

    final_image = result['sample'][0]
    final_image.save('output/' + str(str_image)+'.png')
    return 'output/' + str(str_image)+'.png'

# ### MAIN
#   
if __name__ == "__main__":     
    downloadModel()
    ov_pipe = compileModel()
    imagename = generateImage('xpu', ov_pipe)
    print(imagename)
'''   
    #os.system('pip install -r requirements.txt')
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", force_download=False)
    
    os.mkdir('ovModels')
    
    convert_encoder_ov(pipe)
    convert_unet_ov(pipe)
    convert_vae_ov(pipe)
    
    core = Core()
    text_enc = core.compile_model(TEXT_ENCODER_OV_PATH, 'GPU')
    unet_model = core.compile_model(UNET_OV_PATH, 'GPU')
    vae = core.compile_model(VAE_OV_PATH, 'GPU')
    
    lms = LMSDiscreteScheduler(
        beta_start=0.00085, 
        beta_end=0.012, 
        beta_schedule="scaled_linear"
    )
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')

    ov_pipe = OVStableDiffusionPipeline(
        tokenizer=tokenizer,
        text_encoder=text_enc,
        unet=unet_model,
        vae=vae,
        scheduler=lms
    )    

    text_prompt = 'an old man with a horse'
    num_steps = 50
    seed = random.randint(0, 1024)  
    
    print('Pipeline settings')
    print(f'Input text: {text_prompt}')
    print(f'Seed: {seed}')
    print(f'Number of steps: {num_steps}')
    
    useGif = False
    result = ov_pipe(text_prompt, num_inference_steps=int(num_steps), seed=int(seed), gif=useGif)

    str_image = str(text_prompt) + '_seed_' + str(seed) + '_step_' + str(num_steps)

    final_image = result['sample'][0]
    if useGif:
        all_frames = result['iterations']
        img = next(iter(all_frames))
        img.save(fp='output/gif/'+str(str_image)+'.gif', format='GIF', append_images=iter(all_frames), save_all=True, duration=len(all_frames) * 5, loop=0)
    final_image.save('output/png/'+str(str_image)+'.png')
'''