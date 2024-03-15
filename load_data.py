import os
import json
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset
import math


# val_rate is the sampling rate used to verify that plot_image is the number of plots drawn.
def read_split_data(root: str, val_rate: float = 0.2, plot_image=False):
    random.seed(0)  # Ensure that randomised results are reproducible
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # traverse folders, one folder corresponds to one category, flower_class outputs the category name
    target_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    target_class.sort()  # Sort to ensure consistent order across platforms
    # Generate the category name and the corresponding numeric index, written wonderfully,
    # k is the str name, v is the int numeric number
    class_indices = dict((k, v) for v, k in enumerate(target_class))
    # indent means that characters are indented and line breaks are added after each combination.
    # json.dumps turns dictionaries into strings.
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    for cla in target_class:
        cla_path = os.path.join(root, cla)
        # Iterate over the paths of all files supported by supported, images is a list.
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]  # Get the index corresponding to the category, single image
        every_class_num.append(len(images))  # of samples recorded for the category
        # Proportionate random sampling of validation samples
        val_path = random.sample(images, k=int(len(images) * val_rate))
        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    if plot_image:
        # :: Plotting the number of each category on a bar chart
        plt.bar(range(len(target_class)), every_class_num, align='center')
        # Replace the horizontal coordinates 0,1,2,3,4 with the corresponding category names.
        plt.xticks(range(len(target_class)), target_class)
        # Add numerical labels to bar charts
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


# This function reads the data without dividing the data set, directly read the data set of different months
def read_data(root: str, flag='train', plot_image=True):
    random.seed(0)
    assert os.path.exists(root), "{} dataset root: {} does not exist.".format(flag, root)

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    # Generate the category name and the corresponding numeric index, k is the str name, v is the int numeric number
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    # indent means that characters are indented and line breaks are added after each combination.
    # json.dumps turns dictionaries into strings.
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    images_path = []
    images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".jpeg"]
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        for img_path in images:
            images_path.append(img_path)
            images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for {}.".format(len(images_path), flag))
    assert len(images_path) > 0, "number of images must greater than 0."

    if plot_image:
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        plt.xticks(range(len(flower_class)), flower_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('{} class distribution'.format(flag))
        plt.show()

    return images_path, images_label


class MyDataSet_1(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.collate_fn = None
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        global img2
        img = Image.open(self.images_path[item]).convert('RGB')
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]
        # crop((x0,y0,x1,y1)), x0 is the distance from the left boundary |, y0 is the distance from the upper boundary,
        # x1 is the width of x0+ to be cut off, y1 is the height of y0+ to be cut off
        img_corp = img.crop([0, 0, 64, 64])
        # Note that the 128 here represents the coordinate position, not the actual screenshot width
        img_corp2 = img.crop([64, 0, 128, 64])
        if self.transform is not None:
            img = self.transform(img_corp)
            img2 = self.transform(img_corp2)
        return img, img2, label

    @staticmethod
    def collate_fn(batch):
        images, images2, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        images2 = torch.stack(images2, dim=0)
        labels = torch.as_tensor(labels)
        return images, images2, labels


class MyDataSet_2(Dataset):
    """自定义数据集"""

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


class MyDataSet_3(Dataset):
    """自定义数据集"""

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
        # --------------------------------------------------------------------------------------------------------------
        # Since the design of the spliced figure is a square, the number of pictures is rounded down to the square
        # root of the number of pictures to get the number of small pictures needed for each row of the spliced square.
        perPicNum = math.ceil(math.sqrt(img_quantity))
        # --------------------------------------------------------------------------------------------------------------
        # Generate a fixed size image, similar to the feeling of a canvas, used to paste all the images,
        # and then generate a new image
        toImage = Image.new('RGB', (perPicNum * img_height, perPicNum * img_height), "black")
        # --------------------------------------------------------------------------------------------------------------
        for i in range(img_quantity):
            # fromImage = Image.open(allTransPicPath[i]) # Get the image of the picture used for stitching
            sub_image = img.crop([i * img_height, 0, (i + 1) * img_height, 64])
            # ----------------------------------------------------------------------------------------------------------
            # Calculate the position of each image to ensure smooth splicing, img_height refers to the length and width
            # of the small picture, because the length and width of the same here, so use img_height instead.
            loc = ((int(i / perPicNum) * img_height), (i % perPicNum) * img_height)
            # ----------------------------------------------------------------------------------------------------------
            # print(loc) # Print where each image is located, you can see the distribution
            # Paste the image on the canvas image generated above to the specified position
            toImage.paste(sub_image, loc)
            # ----------------------------------------------------------------------------------------------------------
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
