import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2



import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor


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
    bin_path = args.bin
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    image = cv2.imread(image_path)
    bin_image = cv2.imread(bin_path)
    bin_image = cv2.cvtColor(bin_image, cv2.COLOR_RGB2GRAY)
    pts_fg = np.where(bin_image != 0)

    #1. Define model
    device = "cuda:"+args.gpu
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    n = 1  # for 2 random indices
    if n > 1:
        index = np.random.choice(pts_fg[0].shape[0], n, replace=False)  
        input_point = np.array([    ([j,i])   for i, j in zip(pts_fg[0][index],pts_fg[1][index])])
    else: 
        index = np.random.choice(pts_fg[0].shape[0], n, replace=False)  
        input_point = np.array([[pts_fg[1][index].item(),pts_fg[0][index].item()]])
        
    input_label = np.ones(shape=n)
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask_image = show_mask(mask, image)
        mask_file = "mask_{:.4f}.jpg".format(score)
        image = cv2.imread(image_path)
        if n > 1:
            for j in range(n):
                cv2.drawMarker(image, input_point[j],(0,0,255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2, line_type=cv2.LINE_AA)
        else: 
            cv2.drawMarker(image, input_point[0],(0,0,255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2, line_type=cv2.LINE_AA)        
        mask_image = np.hstack((mask_image, image))
        cv2.imwrite(mask_file, mask_image)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=str, default='0', help="GPU to use")
    parser.add_argument("--bin", type=str, default='/home/khiem/segment-anything/demo_imgs/bin_mask_Chips_chips.png', help="bin_mask")
    parser.add_argument("--img", type=str, default="/home/khiem/segment-anything/demo_imgs/chips.png", help="image")
    args = parser.parse_args()
    main(args)