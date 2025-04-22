from circle_dataset import MyDataset
import cv2

dataset = MyDataset()
print(len(dataset))

item = dataset[123]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)

cv2.imwrite('circleimg.png', jpg)
cv2.imwrite('hintimg.png', hint)