# -*- coding: UTF-8 -*-

from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import torch
from torch import autocast
import os

def downloadModel():
    if not os.path.exists("stable-diffusion-v1-4"):
        os.system("git lfs install")
        os.system("git clone https://huggingface.co/compvis/stable-diffusion-v1-4")
        
def compileModel(xpu):
    if xpu == 'CPU' or xpu == 'GPU':
        lms = LMSDiscreteScheduler.from_pretrained("./stable-diffusion-v1-4", subfolder="scheduler")

    if xpu == 'CPU':
        pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-4", scheduler=lms, torch_dtype=torch.float32, revision="fp32")
        return pipe
    elif xpu == 'GPU':
        pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-4", scheduler=lms, torch_dtype=torch.float16, revision="fp16")
        pipe = pipe.to("cuda")    
        return pipe

def generateImage(xpu, pipe, prompt='horse', negative='', seed=0, steps=20):
    if not os.path.exists('output'):
        os.mkdir('output')

    print('Pipeline settings')
    print(f'Input text: {prompt}')
    print(f'Seed: {seed}')
    print(f'Number of steps: {steps}')
    
    if xpu == 'CPU':
        generator = torch.manual_seed(int(seed))
        image = pipe(prompt, num_inference_steps=int(steps), generator=generator).images[0]
    elif xpu == 'GPU': 
        with autocast("cuda"):
            generator = torch.Generator("cuda").manual_seed(int(seed))
            image = pipe(prompt, negative_prompt=negative, num_inference_steps=int(steps), generator=generator).images[0]

    str_image = str(prompt) + '_seed_' + str(seed) + '_step_' + str(steps)
    image.save('output/' + str(str_image)+'.png')
    return 'output/' + str(str_image)+'.png'