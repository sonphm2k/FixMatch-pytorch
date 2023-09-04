import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from .transformFixMatch import TransformFixMatch

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def get_dataset(root_labeled, root_unlabeled, one_hot_label):
    transform_train_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=200,
                              padding=int(112*0.125),
                              padding_mode='reflect'),
        transforms.Resize(size = (224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean= mean, std=std)
        ])
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean, std=std)
    # ])

    train_labeled_dataset = Labeled_set(one_hot_label=one_hot_label,
                                         root_dir=root_labeled,
                                         transform=transform_train_labeled
                                        )
    train_unlabeled_dataset = Unlabeled_set(root_dir=root_unlabeled,
                                            transform=TransformFixMatch()
                                            )
    return train_labeled_dataset, train_unlabeled_dataset

# One-hot Label
def one_hot_labels(root_dir):
    list_label = []
    one_hot_label = {}
    for label in os.listdir(root_dir):
            list_label.append(label)
    for index, label in enumerate(list_label):
        one_hot_label[label] = index
    return one_hot_label

class Labeled_set(Dataset):
  def __init__(self, root_dir, one_hot_label, transform=None):
    self.list_images_path = []
    self.list_labels = []
    self.one_hot_label = one_hot_label

    for label in os.listdir(root_dir):
        img_dir = os.path.join(root_dir, label)
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            self.list_images_path.append(img_path)
            self.list_labels.append(label)
    self.transform = transform
  def __len__(self):
    return len(self.list_images_path)

  def __getitem__(self, idx):
    label = np.array(self.one_hot_label[self.list_labels[idx]])
    image = cv2.imread(self.list_images_path[idx])
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Numpy -> PIL.Image for transform
    image = Image.fromarray(image)
    if self.transform is not None:
        image = self.transform(image)
    return image, label # tensor, label

class Unlabeled_set(Dataset):
  def __init__(self, root_dir = '', transform=None):
    self.root_dir = root_dir
    self.list_images_name = []
    for image_name in os.listdir(root_dir):
        if '.jpg' in image_name: # Hidden file
            image_path = os.path.join(root_dir, image_name)
            self.list_images_name.append(image_name)
    self.transform = transform

  def __len__(self):
    return len(self.list_images_name)

  def __getitem__(self, idx): # Không dùng được biến đầu vào ở def __it__(..)
    image = cv2.imread(self.root_dir + '/' + self.list_images_name[idx])
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Numpy -> PIL.Image for transform
    image = Image.fromarray(image)
    if self.transform is not None:
        image = self.transform(image) #Tensor(3,h,w)
    image_name = self.list_images_name[idx]
    return image_name, image

# #Test
# root_labeled = './train/Images'
# root_unlabeled = './phase1-test-images'
# one_hot_label = one_hot_labels(root_labeled)
# label, unlabel = get_dataset(root_labeled, root_unlabeled, one_hot_label)
# print(len(label))
# print(len(unlabel))
# for _, __ in label:
#     print(__)