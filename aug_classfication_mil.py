
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage
from numba import jit
import matplotlib.pyplot as plt

ia.seed(1)

def aug_func(img, im_pad_val=-1024):
    # img: h w n
    # lbl: h w
    width_, height_, _ = img.shape
    # if np.random.random()>0.5:
        # width_ = int(width_*1.25)
    # Define our augmentation pipeline.
    seq = iaa.Sequential([
        # horizontal flips
        iaa.Fliplr(0.5),  # horizontal flips
        # iaa.CenterCropToFixedSize(width=int(width_*0.75), height=(height_*0.75)),
        iaa.Sometimes(0.5,
                      iaa.Crop(percent=(0, 0.3)),  # random crops, crop的幅度为0到30%
                      # iaa.CenterCropToFixedSize(width=int(width_ * 0.75), height=int(height_ * 0.75)),
                      ),
        # # Small gaussian blur with random sigma between 0 and 0.5.
        # # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                      ),
        # # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # # Add gaussian noise.
        # # For 50% of all images, we sample the noise once per pixel.
        # # For the other 50% of all images, we sample the noise per pixel AND
        # # channel. This can change the color (not only brightness) of the
        # # pixels.
        # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # # Make some images brighter and some darker.
        # # In 20% of all cases, we sample the multiplier once per channel,
        # # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # # Apply affine transformations to each image.
        # # Scale/zoom them, translate/move them, rotate them and shear them.

        iaa.Sometimes(0.5,
            iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            rotate=(-90, 90),
            shear=(-8, 8),
            mode=["constant"],
            cval=im_pad_val,
                ),
                      )
        # iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
        # iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
    ], random_order=True)

    img_aug = seq(images=[img])



    return img_aug[0]

# @jit
def aug_process(imgs, im_pad_val=-1024):
    # imgs: (h,w, c)

    # imgs = np.transpose(imgs, (0, 2, 3, 1))  # (N,h,w,cls)
    images_aug = aug_func(imgs, im_pad_val=im_pad_val)
    # images_aug = np.transpose(images_aug, (0, 3, 1, 2))  # (N,h,w,cls)
    return images_aug




