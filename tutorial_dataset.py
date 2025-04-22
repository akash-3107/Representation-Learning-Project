import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from annotator.util import resize_image, HWC3
import tifffile as tiff

class CityscapesDataset(Dataset):
    def __init__(self, split='train', low_threshold=100, high_threshold=200):
        self.root_dir = '/home/hpc/rlvl/rlvl129v/ControlNet/data'
        self.split = split
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        # Define paths for images and segmentation masks
        self.image_dir = os.path.join(self.root_dir, 'leftImg8bit', split)
        self.mask_dir = os.path.join(self.root_dir, 'gtFine', split)

        # Collect image paths
        self.images = []
        for city in os.listdir(self.image_dir):
            city_path = os.path.join(self.image_dir, city)
            for file in os.listdir(city_path):
                if file.endswith("_leftImg8bit.png"):
                    image_path = os.path.join(city_path, file)
                    mask_path = os.path.join(
                        self.mask_dir, city, file.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                    )
                    if os.path.exists(mask_path):
                        self.images.append((image_path, mask_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, mask_path = self.images[idx]
        # Load image and segmentation mask
        target = cv2.imread(image_path)
        source = cv2.imread(mask_path)

        # Convert BGR to RGB
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        

        '''
        (B, G, R) = cv2.split(target_rgb)
        B_cny = cv2.Canny(B, self.low_threshold, self.high_threshold)
        G_cny = cv2.Canny(G, self.low_threshold, self.high_threshold)
        R_cny = cv2.Canny(R, self.low_threshold, self.high_threshold)
        canny = cv2.merge([B_cny, G_cny, R_cny])

        #canny = cv2.Canny(target, self.low_threshold, self.high_threshold)
        #canny = np.expand_dims(canny, axis=2)
        '''

        #detected_map = cv2.Canny(target_rgb, self.low_threshold, self.high_threshold)
        #canny = HWC3(detected_map)

        #source = np.concatenate((source, canny), axis=2)
        
        # Normalize source image to [0, 1]
        source = source.astype(np.float32) / 255.0

        # Normalize target to [0, 1] and keep label IDs
        target = (target.astype(np.float32) / 127.5) - 1.0
        #cv2.imwrite('source.png', source)
        #cv2.imwrite('target.png', target)
        prompt = ""

        return dict(jpg=target, txt=prompt, hint=source)
