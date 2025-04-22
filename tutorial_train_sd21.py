from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import CityscapesDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch

def modify_state_dict_for_6_channels(state_dict):
    for key in state_dict.keys():
        if 'input_hint_block' in key and 'weight' in key:
            old_weight = state_dict[key]
            if old_weight.shape[1] == 3:
                new_weight = torch.zeros(old_weight.shape[0], 6, old_weight.shape[2], old_weight.shape[3], 
                                         device=old_weight.device, dtype=old_weight.dtype)
                new_weight[:, :3, :, :] = old_weight
                new_weight[:, 3:, :, :] = old_weight  # Duplicate the weights for the additional channels
                state_dict[key] = new_weight
    return state_dict


# Configs
resume_path = './models/control_sd21_ini.ckpt'
batch_size = 2
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
max_epochs = 3

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()

state_dict = load_state_dict(resume_path, location='cpu')
#state_dict = modify_state_dict_for_6_channels(state_dict)

repeated_tensor = torch.stack([torch.tensor(item).repeat(1, 3, 1, 1) for item in state_dict['control_model.input_hint_block.0.weight']]).squeeze(1)

# Assign the corrected tensor to the state dictionary
state_dict['control_model.input_hint_block.0.weight'] = repeated_tensor

model.load_state_dict(state_dict)

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = CityscapesDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], max_epochs = max_epochs)


# Train!
trainer.fit(model, dataloader)
