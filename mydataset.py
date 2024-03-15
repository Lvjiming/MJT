from PIL import Image
import torch
from torch.utils.data import Dataset
import math


class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert('RGB')
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


class MyDataSet_2(Dataset):
    # The input is a horizontally stitched multi-view image.
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert('RGB')
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        img_width = img.size[0]
        img_height = img.size[1]
        # Since the images are stitched horizontally and are all squares,
        # divide the width by the height to get the number of images
        img_quantity = math.ceil(img_width / img_height)
        # perPicNum = math.ceil(math.sqrt(img_quantity))
        # toImage = Image.new('RGB', (perPicNum * img_height, perPicNum * img_height),"black")
        # toImage.show()
        eval_image = []
        for i in range(img_quantity):
            # fromImage = Image.open(allTransPicPath[i])
            sub_image = img.crop([i * img_height, 0, (i + 1) * img_height, 64])
            # Calculate the position of each image to ensure smooth splicing, img_height refers to the length
            # and width of the small picture, because the length and width of the same here, so use img_height instead.
            # loc = ((int(i / perPicNum) * img_height), (i % perPicNum) * img_height)
            # Print where each image is located, you can see the distribution
            # print(loc)
            # Paste the image on the canvas image generated above to the specified position
            # toImage.paste(sub_image, loc)
            # toImage.show()
            # if img_quantity == 1:
            #     new_img = img
            # else 1 < img_quantity <= 4:
            # img_corp = img.crop([0, 0, 64, 64])
            eval_image.append(sub_image)
        images = []
        if self.transform is not None:
            for j in range(len(eval_image)):
                sub_trans_image = self.transform(eval_image[j])
                images.append(sub_trans_image)
        return images, label

    @staticmethod
    # The input to this function is a list whose length is a batch size.
    # # Each element of the list is the result of __getitem__.
    def collate_fn(dataset):
        # images, labels = tuple(zip(*batch)
        images, labels = zip(*dataset)
        images_all = []
        label_all = []
        view_num = len(images[0])
        for j in range(len(images)):
            for j2 in range(view_num):
                images_all.append(images[j][j2])
        for j3 in range(len(labels)):
            for j4 in range(view_num):
                label_all.append(labels[j3])  # With a few perspectives, add a few tags #
        images = torch.stack(images_all, dim=0)

        labels_mutil_view = torch.as_tensor(label_all)
        # This single view means that the multi-view labels have been merged, and the number of labels remains
        # the same as in the original batch-size.
        labels_single = torch.as_tensor(labels)
        return images, labels_single, labels_mutil_view, view_num


# MyDataSet_3 is a horizontally stitched multi-view image, decomposed, then merged into a square
class MyDataSet_3(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert('RGB')
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        img_width = img.size[0]
        img_height = img.size[1]
        img_quantity = math.ceil(img_width / img_height)

        perPicNum = math.ceil(math.sqrt(img_quantity))
        toImage = Image.new('RGB', (perPicNum * img_height, perPicNum * img_height), "black")
        # toImage.show()
        for i in range(img_quantity):
            sub_image = img.crop([i * img_height, 0, (i + 1) * img_height, 64])
            loc = ((int(i / perPicNum) * img_height), (i % perPicNum) * img_height)
            toImage.paste(sub_image, loc)
        if self.transform is not None:
            img = self.transform(toImage)
        # AA = 1
        return img, label

    @staticmethod
    def collate_fn(dataset):
        images, labels = zip(*dataset)
        images_all = []
        label_all = []
        view_num = len(images[0])
        for j in range(len(images)):
            for j2 in range(view_num):
                images_all.append(images[j][j2])
        for j3 in range(len(labels)):
            for j4 in range(view_num):
                label_all.append(labels[j3])

        images = torch.stack(images_all, dim=0)
        labels_mutil_view = torch.as_tensor(label_all)
        labels_single = torch.as_tensor(labels)
        return images, labels_single, labels_mutil_view, view_num


class MyDataSet_4(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert('RGB')
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        img_width = img.size[0]
        img_height = img.size[1]
        img_quantity = math.ceil(img_width / img_height)

        eval_image = []
        for i in range(img_quantity):
            sub_image = img.crop([i * img_height, 0, (i + 1) * img_height, img_height])

            eval_image.append(sub_image)
        images = []
        if self.transform is not None:
            for j in range(len(eval_image)):
                sub_trans_image = self.transform(eval_image[j])
                images.append(sub_trans_image)

        AA = 4
        return images, label

    @staticmethod
    def collate_fn(dataset):
        images, labels = zip(*dataset)
        images_all = []
        label_all = []
        view_num = len(images[0])
        for j in range(len(images)):
            for j2 in range(view_num):
                images_all.append(images[j][j2])
        for j3 in range(len(labels)):
            for j4 in range(view_num):
                label_all.append(labels[j3])

        images = torch.stack(images_all, dim=0)

        labels_mutil_view = torch.as_tensor(label_all)
        labels_single = torch.as_tensor(labels)
        return images, labels_single, labels_mutil_view, view_num
