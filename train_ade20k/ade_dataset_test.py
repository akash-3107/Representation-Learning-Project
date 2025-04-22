from ade_dataset import ADEDataset
import cv2

dataset = ADEDataset()
print(len(dataset))


item = dataset[123]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']


print(txt)
print(jpg.shape)
print(hint.shape)
