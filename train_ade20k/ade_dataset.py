import json
import cv2
import numpy as np
import sys
sys.path.append("..")
from torch.utils.data import Dataset
from annotator.util import resize_image, HWC3


class ADEDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('/home/hpc/rlvl/rlvl129v/ControlNet/training/ade20k/ade_prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        t_lower = 100
        t_upper = 100

        item = self.data[idx]

        # source is image, target is mask. switch required
        img_path = item['source']
        mask_path = item['target']
        prompt = item['prompt']
        prompt += ' with a filled circle in the image'
        #prompt = ''
        resolution = 512

        source = cv2.imread(mask_path)
        target = cv2.imread(img_path)

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        source = cv2.resize(source, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
        target = cv2.resize(target, (resolution, resolution), interpolation=cv2.INTER_LINEAR)

        center = (resolution // 2, resolution // 2)
        radius = 50
        cv2.circle(target, center, radius, (255), thickness=-1)

        '''
        canny = cv2.Canny(target, t_lower, t_upper)
        canny = HWC3(canny)
        source = np.concatenate((source, canny), axis=2)
        '''
        
        #'''
        #additional_hint = cv2.imread('/home/hpc/rlvl/rlvl129v/ControlNet/test_imgs/dog2.png')
        additional_hint = cv2.imread('/home/hpc/rlvl/rlvl129v/ControlNet/training/fill50k/target/89.png')
        additional_hint = cv2.cvtColor(additional_hint, cv2.COLOR_BGR2RGB)
        additional_hint = cv2.resize(additional_hint, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
        canny = cv2.Canny(additional_hint, t_lower, t_upper)
        canny = HWC3(canny)
        source = np.concatenate((source, canny), axis=2)
        #'''

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
