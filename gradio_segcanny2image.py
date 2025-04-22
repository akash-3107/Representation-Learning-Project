from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


apply_uniformer = UniformerDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('/home/woody/rlvl/rlvl129v/lightning_logs_ade20k/version_1013990_6channels/checkpoints/epoch=5-step=19181.ckpt', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image1, input_image2, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    with torch.no_grad():
        input_image1 = HWC3(input_image1)
        detected_map1 = apply_uniformer(resize_image(input_image1, detect_resolution))
        img1 = resize_image(input_image1, image_resolution)
        H1, W1, C1 = img1.shape
        detected_map1 = cv2.resize(detected_map1, (W1, H1), interpolation=cv2.INTER_NEAREST)


        img2 = resize_image(HWC3(input_image2), image_resolution)
        H2, W2, C2 = img2.shape
        detected_map2 = cv2.Canny(img2, low_threshold, high_threshold)
        detected_map2 = HWC3(detected_map2)

        print(detected_map1.shape)
        print(detected_map2.shape)

        detected_map = np.concatenate((detected_map1, detected_map2), axis=2)
        
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        
        
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        #if config.save_memory:
        #    model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H1 // 8, W1 // 8)

        #if config.save_memory:
        #    model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        #if config.save_memory:
        #    model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        
    return [detected_map] + results


prompt = 'a dog with a hat'
a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
num_samples = 6
image_resolution = 512
detect_resolution = 512
ddim_steps = 40
guess_mode = False
strength = 1.0
scale = 9.0
seed = random.randint(-1, 2147483647)
eta = 0.0
low_threshold = 100
high_threshold = 200


input_image1 = cv2.imread('/home/hpc/rlvl/rlvl129v/ControlNet/test_imgs/dog2.png')
input_image1 = cv2.cvtColor(input_image1, cv2.COLOR_BGR2RGB)
input_image2 = input_image1
res = process(input_image1, input_image2, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold)


print(len(res))
#print(type(res))

#print(res[0].shape)

for r in range(len(res)):
    print(res[r][:,:,:3].shape)
    print(res[r][:,:,3:].shape)
    print('------------')
    #cv2.imwrite('out_seg_canny_' + str(r) + '_1' + '.png', res[r][:,:,:3])
    #cv2.imwrite('out_seg_canny_' + str(r) + '_2' + '.png', res[r][:,:,3:])

