from torchvision import transforms
from .randaugment import RandAugmentMC

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

class TransformFixMatch(object):
    def __init__(self):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=200,
                                padding=int(112*0.125),
                                padding_mode='reflect'),
        ])

        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=200,
                                  padding=int(112*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=3, m=10),
        ])

        self.normalize = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean= mean, std= std),
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)