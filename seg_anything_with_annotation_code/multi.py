import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


import sys
sys.path.append("..")
from segment_anything import sam_model_registry,  SamAutomaticMaskGenerator, SamPredictor


def show_mask(mask, img, random_color=False, opacity=0.6):
    if random_color:
        color = np.concatenate([np.random.random(3)*255], axis=0)
    else:
        color = np.array([30, 144, 255])
    h, w = mask.shape[-2:]
    color_seg = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    fg_mask = mask != False
    
    img[fg_mask] = color_seg[fg_mask] * opacity
    
    return img
    



def main(args):
    image_path = args.img
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    image = cv2.imread(image_path)

    #1. Define model
    device = "cuda:"+args.gpu
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    anns = mask_generator.generate(image)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    opacity=0.5
    for ann in sorted_anns:
        mask= ann['segmentation']
        color = np.concatenate([np.random.random(3)*255], axis=0)
        h, w = mask.shape[-2:]
        color_seg = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        fg_mask = mask != False
        image[fg_mask] = color_seg[fg_mask] * opacity
    image_orig = cv2.imread(image_path)
    image = np.hstack((image_orig, image))
    cv2.imwrite("multi_mask.jpg", image)

        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=str, default='0', help="GPU to use")
    parser.add_argument("--img", type=str, default="/home/khiem/segment-anything/demo_imgs/multi_obj.jpg", help="image")
    args = parser.parse_args()
    main(args)