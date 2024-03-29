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
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, EulerAncestralDiscreteScheduler
from openvino.runtime import Model
from openvino.runtime import Core
from transformers import CLIPTokenizer
#import ipywidgets as widgets
import random
import time

from typing import List, Optional, Union, Dict
import PIL
import cv2

TEXT_ENCODER_ONNX_PATH = Path('ovTempModels/text_encoder.onnx')
TEXT_ENCODER_OV_PATH = Path('ovModels/text_encoder.xml')
UNET_ONNX_PATH = Path('ovTempModels/unet.onnx')
UNET_OV_PATH = Path('ovModels/unet.xml')
VAE_ENCODER_ONNX_PATH = Path('ovTempModels/vae_encoder.onnx')
VAE_ENCODER_OV_PATH = Path('ovModels/vae_encoder.xml')
VAE_DECODER_ONNX_PATH = Path('ovTempModels/vae_decoder.onnx')
VAE_DECODER_OV_PATH = Path('ovModels/vae_decoder.xml')

# ### Text Encoder
# 
# The text-encoder is responsible for transforming the input prompt, e.g. "a photo of an astronaut riding a horse" into an embedding space that can be understood by the U-Net. It is usually a simple transformer-based encoder that maps a sequence of input tokens to a sequence of latent text embeddings.
# 
# Input of text encoder is tensor `input_ids` which contains indexes of tokens from text processed by tokenizer and padded to maximum length accepted by model. Model outputs are 2 tensors: `last_hidden_state` - hidden state from the last MultiHeadAttention layer in model and `pooler_out` - Pooled output for whole model hidden states. We will use `opset_version=14`, because model contains `triu` operation, supported in ONNX only starting from this opset.
def convert_encoder_onnx(text_encoder: StableDiffusionPipeline, onnx_path:Path):
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
        input_ids = torch.ones((1, 77), dtype=torch.long)
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
                input_names=['tokens'],  # model input name for onnx representation
                output_names=['last_hidden_state', 'pooler_out'],  # model output names for onnx representation
                opset_version=14  # onnx opset version for export
            )
        print('Text Encoder successfully converted to ONNX')
    else:
        print('Text Encoder exists for ONNX')
    
def convert_encoder_ov(text_encoder):
    if not TEXT_ENCODER_OV_PATH.exists():
        convert_encoder_onnx(text_encoder, TEXT_ENCODER_ONNX_PATH)
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
def convert_unet_onnx(unet:StableDiffusionPipeline, onnx_path:Path):
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
        encoder_hidden_state = torch.ones((2, 77, 768))
        latents_shape = (2, 4, 512 // 8, 512 // 8)
        latents = torch.randn(latents_shape)
        t = torch.from_numpy(np.array(1, dtype=float))

        # model size > 2Gb, it will be represented as onnx with external data files, we will store it in separated directory for avoid a lot of files in current directory
        onnx_path.parent.mkdir(exist_ok=True, parents=True)
        unet.eval()

        with torch.no_grad():
            torch.onnx.export(
                unet, 
                (latents, t, encoder_hidden_state), str(onnx_path),
                input_names=['latent_model_input', 't', 'encoder_hidden_states'],
                output_names=['out_sample']
            )
        print('Unet successfully converted to ONNX')
    else:
        print('Unet exists for ONNX')

def convert_unet_ov(unet):
    if not UNET_OV_PATH.exists():
        convert_unet_onnx(unet, UNET_ONNX_PATH)
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
def convert_vae_encoder_onnx(vae:StableDiffusionPipeline, onnx_path:Path):
    """
    Convert VAE model to ONNX, then IR format. 
    Function accepts pipeline, creates wrapper class for export only necessary for inference part, 
    prepares example inputs for ONNX conversion via torch.export, 
    Parameters: 
        pipe (StableDiffusionInstructPix2PixPipeline): InstrcutPix2Pix pipeline
        onnx_path (Path): File for storing onnx model
    Returns:
        None
    """
    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, image):
            h = self.vae.encoder(image)
            moments = self.vae.quant_conv(h)
            return moments

    if not onnx_path.exists():
        vae_encoder = VAEEncoderWrapper(vae)
        vae_encoder.eval()
        image = torch.zeros((1, 3, 512, 512))
        with torch.no_grad():
            torch.onnx.export(vae_encoder, image, onnx_path, input_names=[
                              'init_image'], output_names=['image_latent'])
        print('VAE encoder successfully converted to ONNX')
    else:
        print('VAE encoder exists for ONNX')
        
def convert_vae_encoder_ov(vae):
    if not VAE_ENCODER_OV_PATH.exists():
        convert_vae_encoder_onnx(vae, VAE_ENCODER_ONNX_PATH)
        os.system("mo --input_model " + VAE_ENCODER_ONNX_PATH.parent.name + "/" + VAE_ENCODER_ONNX_PATH.name + " --output_dir " + VAE_ENCODER_OV_PATH.parent.name + " --compress_to_fp16")
        print('VAE encoder successfully converted to IR')
    else:
        print('VAE encoder exists for IR')        

def convert_vae_decoder_onnx(vae:StableDiffusionPipeline, onnx_path:Path):
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
            latents = 1 / 0.18215 * latents # community moved the factor from vae outside to inside
            return self.vae.decode(latents)

    if not onnx_path.exists():
        vae_decoder = VAEDecoderWrapper(vae)
        latents = torch.zeros((1, 4, 64, 64))

        vae_decoder.eval()
        with torch.no_grad():
            torch.onnx.export(vae_decoder, latents, onnx_path, input_names=[
                              'latents'], output_names=['sample'])
        print('VAE decoder successfully converted to ONNX')
    else:
        print('VAE decoder exists for ONNX')

def convert_vae_decoder_ov(vae):
    if not VAE_DECODER_OV_PATH.exists():
        convert_vae_decoder_onnx(vae, VAE_DECODER_ONNX_PATH)
        os.system("mo --input_model " + VAE_DECODER_ONNX_PATH.parent.name + "/" + VAE_DECODER_ONNX_PATH.name + " --output_dir " + VAE_DECODER_OV_PATH.parent.name + " --compress_to_fp16")
        print('VAE decoder successfully converted to IR')
    else:
        print('VAE decoder exists for IR')

######################################################################

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
        vae_encoder: Model,
        vae_decoder: Model,
        text_encoder: Model,
        tokenizer: CLIPTokenizer,
        unet: Model,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, EulerAncestralDiscreteScheduler],
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
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.text_encoder = text_encoder
        self.unet = unet
        self._text_encoder_output = text_encoder.output(0)
        self._unet_output = unet.output(0)
        self._vae_e_output = vae_encoder.output(0) if vae_encoder is not None else None
        self._vae_d_output = vae_decoder.output(0)
        self.height = self.unet.input(0).shape[2] * 8
        self.width = self.unet.input(0).shape[3] * 8
        self.tokenizer = tokenizer

    @torch.no_grad()
    def __call__(
        self,
        prompt: [str],
        negative_prompt: [str],
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        image: PIL.Image.Image = None,
        strength: Optional[float] = 0.5,
        output_type: Optional[str] = "pil",
        seed: Optional[int] = None,
        mask: Optional[any] = None,
        gif: Optional[bool] = False,
        callback: Optional[any] = None,
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
            image (PIL.Image.Image, *optional*, None):
                 Intinal image for generation.    
            strength:
                Strength of the noise in image to image. 
            output_type (`str`, *optional*, defaults to "pil"):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): PIL.Image.Image or np.array.
            seed (int, *optional*, None):
                Seed for random generator state initialization.
            mask:
                Mask for inpaint image to image
            gif (bool, *optional*, False):
                Flag for storing all steps results or not.
            callback:
                Callback once after a step. 
        Returns:
            Dictionary with keys: 
                sample - the last generated image PIL.Image.Image or np.array
                iterations - *optional* (if gif=True) images for all diffusion steps, List of PIL.Image.Image or np.array.
        """

        if seed is not None:
            np.random.seed(seed)
            generator = torch.manual_seed(seed)

        if isinstance(prompt, str):
            batch_size = 1
        else:
            raise ValueError(f"`prompt` has to be of type `str` but is {type(prompt)}")

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
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                negative_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="np"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[self._text_encoder_output]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        timesteps = self.scheduler.timesteps        

        # get the initial random noise unless the user supplied it
        latents_shape = (batch_size, 4, self.height // 8, self.width // 8)
       
        # initialize first latents
        noise = np.random.randn(*latents_shape)
        num_modify_steps = num_inference_steps + offset
        if image is None:
            latents = noise
            meta = {}
        else:
            num_modify_steps = int(num_inference_steps * strength) + offset
            num_modify_steps = min(num_modify_steps, num_inference_steps)
            ts = np.array([timesteps[-num_modify_steps]])
            init_latents, latents, meta = self.prepare_latents(image, noise, ts)
            init_latents = np.array(init_latents)
            latents = np.array(latents)
            # handle mask
            #if mask is not None:
            
            
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

        t_start = max(num_inference_steps - num_modify_steps + offset, 0)
        for i, t in enumerate(self.progress_bar(timesteps)):
            # jump over the first several steps when image to image
            if i < t_start:
                # callback each step
                if callback is not None:
                    callback()
                continue
                
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet([latent_model_input, t, text_embeddings])[self._unet_output]
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            #torch.save(noise_pred, str(i)+"_nsp.np")
            #torch.save(latents, str(i)+"_lts.np")

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents), generator=generator, **extra_step_kwargs)["prev_sample"].numpy()
            
            # add mask
            if image is not None and mask is not None:
                ts = np.array([timesteps[i]])
                init_latents_proper = self.add_noise(init_latents, noise, ts)
                latents = ((init_latents_proper * mask) + (latents * (1 - mask)))
            
            # save for gif if needed
            if gif:
                image_output = self.vae_decoder(latents)[self._vae_d_output]
                image_output = self.postprocess_image(image_output, meta, output_type)
                img_buffer.extend(image_output)
                
            # callback each step
            if callback is not None:
                callback()

        # scale and decode the image latents with vae
        #latents = 1 / 0.18215 * latents    # community moved the 0.18215 into vae decoder forward code
        image_output = self.vae_decoder(latents)[self._vae_d_output]
        image_output = self.postprocess_image(image_output, meta, output_type)

        return {"sample": image_output, 'iterations': img_buffer}

    #---------------------------------------------------------------------------------------------------------
    def add_noise(self, original_samples, noise, timesteps):
        scheduler = self.scheduler
        schedule_timesteps = scheduler.timesteps
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = scheduler.sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
        noisy_samples = original_samples + noise * np.array(sigma)
        return noisy_samples
            
    
    def scale_fit_to_window(self, dst_width:int, dst_height:int, image_width:int, image_height:int):
        """
        Preprocessing helper function for calculating image size for resize with peserving original aspect ratio 
        and fitting image to specific window size
        
        Parameters:
          dst_width (int): destination window width
          dst_height (int): destination window height
          image_width (int): source image width
          image_height (int): source image height
        Returns:
          result_width (int): calculated width for resize
          result_height (int): calculated height for resize
        """
        im_scale = min(dst_height / image_height, dst_width / image_width)
        return int(im_scale * image_width), int(im_scale * image_height)


    def preprocess(self, image: PIL.Image.Image):
        """
        Image preprocessing function. Takes image in PIL.Image format, resizes it to keep aspect ration and fits to model input window 512x512,
        then converts it to np.ndarray and adds padding with zeros on right or bottom side of image (depends from aspect ratio), after that
        converts data to float32 data type and change range of values from [0, 255] to [-1, 1], finally, converts data layout from planar NHWC to NCHW.
        The function returns preprocessed input tensor and padding size, which can be used in postprocessing.
        
        Parameters:
          image (PIL.Image.Image): input image
        Returns:
           image (np.ndarray): preprocessed image tensor
           meta (Dict): dictionary with preprocessing metadata info
        """
        src_width, src_height = image.size
        dst_width, dst_height = self.scale_fit_to_window(
            512, 512, src_width, src_height)
        image = np.array(image.resize((dst_width, dst_height),
                         resample=PIL.Image.Resampling.LANCZOS))[None, :]
        pad_width = 512 - dst_width
        pad_height = 512 - dst_height
        pad = ((0, 0), (0, pad_height), (0, pad_width), (0, 0))
        image = np.pad(image, pad, mode="constant")
        image = image.astype(np.float32) / 255.0
        image = 2.0 * image - 1.0
        image = image.transpose(0, 3, 1, 2)
        return image, {"padding": pad, "src_width": src_width, "src_height": src_height}

    def prepare_latents(self, image:PIL.Image.Image, noise, latent_timestep:torch.Tensor):
        """
        Function for getting initial latents for starting generation
        
        Parameters:
            image (PIL.Image.Image, *optional*, None):
                Input image for generation, if not provided randon noise will be used as starting point
            latent_timestep (torch.Tensor, *optional*, None):
                Predicted by scheduler initial step for image generation, required for latent image mixing with nosie
        Returns:
            latents (np.ndarray):
                Image encoded in latent space
        """
        latents_shape = (1, 4, self.height // 8, self.width // 8)
        #noise = np.random.randn(*latents_shape).astype(np.float32)
        if image is None:
            # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                noise = noise * self.scheduler.sigmas[0].numpy()
            return noise, {}
        input_image, meta = self.preprocess(image)
        moments = self.vae_encoder(input_image)[self._vae_e_output]
        mean, logvar = np.split(moments, 2, axis=1) 
        std = np.exp(logvar * 0.5)
        init_latents = (mean + std * np.random.randn(*mean.shape)) * 0.18215
        #latents = self.scheduler.add_noise(torch.from_numpy(latents), torch.from_numpy(noise), latent_timestep).numpy()
        latents = self.add_noise(init_latents, noise, latent_timestep)
        return init_latents, latents, meta

    def postprocess_image(self, image:np.ndarray, meta:Dict, output_type:str = "pil"):
        """
        Postprocessing for decoded image. Takes generated image decoded by VAE decoder, unpad it to initila image size (if required), 
        normalize and convert to [0, 255] pixels range. Optionally, convertes it from np.ndarray to PIL.Image format
        
        Parameters:
            image (np.ndarray):
                Generated image
            meta (Dict):
                Metadata obtained on latents preparing step, can be empty
            output_type (str, *optional*, pil):
                Output format for result, can be pil or numpy
        Returns:
            image (List of np.ndarray or PIL.Image.Image):
                Postprocessed images
        """
        if "padding" in meta:
            pad = meta["padding"]
            (_, end_h), (_, end_w) = pad[1:3]
            h, w = image.shape[2:]
            unpad_h = h - end_h
            unpad_w = w - end_w
            image = image[:, :, :unpad_h, :unpad_w]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = np.transpose(image, (0, 2, 3, 1))
        # 9. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            if "src_height" in meta:
                orig_height, orig_width = meta["src_height"], meta["src_width"]
                image = [img.resize((orig_width, orig_height),
                                    PIL.Image.Resampling.LANCZOS) for img in image]
        else:
            if "src_height" in meta:
                orig_height, orig_width = meta["src_height"], meta["src_width"]
                image = [cv2.resize(img, (orig_width, orig_width))
                         for img in image]
        return image
    '''
    def get_timesteps(self, num_inference_steps:int, strength:float):
        """
        Helper function for getting scheduler timesteps for generation
        In case of image-to-image generation, it updates number of steps according to strength
        
        Parameters:
           num_inference_steps (int):
              number of inference steps for generation
           strength (float):
               value between 0.0 and 1.0, that controls the amount of noise that is added to the input image. 
               Values that approach 1.0 allow for lots of variations but will also produce images that are not semantically consistent with the input.
        """
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start 
    '''


import re
def validateTitle(title):
    rstr = r"[\/\\\:\*\?\"\<\>\|.]"
    return re.sub(rstr, r"", title)

# ### EXPORT
#
#REPO = "darkstorm2150/Protogen_v5.3_Official_Release"
REPO = "windwhinny/chilloutmix"
#REPO = "compvis/stable-diffusion-v1-4"
#REPO = "runwayml/stable-diffusion-v1-5"

def downloadModel():
    if not TEXT_ENCODER_OV_PATH.exists() or not UNET_OV_PATH.exists() or not VAE_DECODER_OV_PATH.exists():
        if not os.path.exists('ovTempModels'):
            os.mkdir('ovTempModels')
        if not os.path.exists('ovModels'):
            os.mkdir('ovModels')    
        pipe = StableDiffusionPipeline.from_pretrained(REPO, force_download=False)
        text_encoder = pipe.text_encoder
        text_encoder.eval()
        unet = pipe.unet
        unet.eval()
        vae = pipe.vae
        vae.eval()
        del pipe
        convert_encoder_ov(text_encoder)
        del text_encoder
        convert_unet_ov(unet)
        del unet
        convert_vae_encoder_ov(vae)   
        convert_vae_decoder_ov(vae)
        del vae

    
def compileModel(xpu):
    if 'CPU' in xpu or 'GPU' in xpu or 'AUTO' in xpu:
        core = Core()
        text_enc = core.compile_model(TEXT_ENCODER_OV_PATH, xpu)
        unet_model = core.compile_model(UNET_OV_PATH, xpu)
        vae_encoder = core.compile_model(VAE_ENCODER_OV_PATH, xpu)
        vae_decoder = core.compile_model(VAE_DECODER_OV_PATH, xpu)
        
        '''
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, 
            beta_end=0.012, 
            beta_schedule="scaled_linear"
        )
        tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
        '''

        try:
            #scheduler = LMSDiscreteScheduler.from_pretrained(REPO, subfolder="scheduler")
            scheduler = EulerAncestralDiscreteScheduler.from_pretrained(REPO, subfolder="scheduler")
            tokenizer = CLIPTokenizer.from_pretrained(REPO, subfolder="tokenizer")
        except:
            #scheduler = LMSDiscreteScheduler.from_pretrained('ovModels', subfolder="scheduler")
            scheduler = EulerAncestralDiscreteScheduler.from_pretrained('ovModels', subfolder="scheduler")
            tokenizer = CLIPTokenizer.from_pretrained('ovModels', subfolder="tokenizer")              

        ov_pipe = OVStableDiffusionPipeline(
            tokenizer=tokenizer,
            text_encoder=text_enc,
            unet=unet_model,
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            scheduler=scheduler
        )        
        return ov_pipe

def generateImage(xpu, ov_pipe, prompt='horse', negative='', seed=0, steps=20, image="", strength=0.5, mask=None, callback=None):
    if not os.path.exists('output'):
        os.mkdir('output')

    print('Pipeline settings')
    print(f'Input text: {prompt}')
    print(f'Negative text: {negative}')
    print(f'Seed: {seed}')
    print(f'Number of steps: {steps}')
    
    try:
        input_image = PIL.Image.open(image)
    except:
        input_image = None
    
    result = ov_pipe(prompt, negative_prompt=negative, num_inference_steps=int(steps), seed=int(seed), image=input_image, strength=strength, mask=mask, gif=False, callback=callback)

    str_image = validateTitle(str(prompt))[:100] + '_seed_' + str(seed) + '_step_' + str(steps)+'_time_'+str(int(time.time()))

    final_image = result['sample'][0]
    final_image.save('output/' + str(str_image)+'.png')
    
    return 'output/' + str(str_image)+'.png'
