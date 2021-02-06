from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    Rotate,
    HueSaturationValue,
	Cutout
)

from albumentations.pytorch import ToTensor
import numpy as np


def albumentations_train_transforms(mean,std,p=1.0):
    
    transforms_list = []
    
    transforms_list.extend([
        HueSaturationValue(p=0.25),
        HorizontalFlip(p=0.5),
        Rotate(limit=15),
		Cutout(num_holes=4),
        Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensor()

    ])

    transforms = Compose(transforms_list, p=p)
    return lambda img: transforms(image=np.array(img))["image"]
