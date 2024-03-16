import torch
from PIL import Image

img_path = 'image path'
subimg_path = '~/subimg/'

img = Image.open(img_path).convert('RGB')
img_tensor = torch.Tensor(list(img.getdata())).view(64, 64, 3).permute(2, 0, 1)

subimg_size = 16

for i in range(4):
    for j in range(4):
        subimg = img_tensor[:, i * subimg_size:(i + 1) * subimg_size, j * subimg_size:(j + 1) * subimg_size]
        subimg = subimg.permute(1, 2, 0).numpy().astype('uint8')
        subimg = Image.fromarray(subimg)
        subimg.save(subimg_path + str(i * 4 + j + 1) + '.jpg')
