from tutorial_dataset import CityscapesDataset
import cv2

dataset = CityscapesDataset()
print(len(dataset))

item = dataset[123]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)

#cv2.imwrite('temp_orig.png', jpg)

#cv2.imwrite('hint1.png', hint[:,:,:3])
#cv2.imwrite('hint.png', hint)
#print(jpg)
#print(hint)