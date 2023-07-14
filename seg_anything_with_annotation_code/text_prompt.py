import clip
import argparse
import torch
import cv2
import os
import numpy as np
from PIL import Image
from  matplotlib import pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

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
    device = "cpu"
    image_path = args.img
    pil_img = Image.open(image_path) 
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    preprocess =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    image = preprocess(pil_img).unsqueeze(0).to(device)

    #1. Init Text+Image Prompters
    model, preprocess = clip.load("CS-ViT-B/16", device=device)
    model.eval()

    #2. Init SAM
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(np.array(pil_img))
    
    #4. CLIP: Get Points
    target_texts = ["screwdriver", "red and black handle screwdriver", "blue and black handle screwdriver"]
                 
    with torch.no_grad():
        # clip architecture surgery acts on the image encoder
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # prompt ensemble for text features with normalization
        text_features = clip.encode_text_with_prompt_ensemble(model, target_texts, device)

        # apply feature surgery, no batch
        similarity = clip.clip_feature_surgery(image_features, text_features)[0]

        similarity_feat = clip.clip_feature_surgery(image_features, text_features)
        similarity_map = clip.get_similarity_map(similarity_feat[:, 1:, :], cv2_img.shape[:2]).squeeze(0)
        
        # inference SAM with points from CLIP Surgery
        for n in range(similarity.shape[-1]):

            points, labels = clip.similarity_map_to_points(similarity[1:, n], cv2_img.shape[:2], t=0.99)
            
            masks, scores, logits = predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True)
            
            for i, mask in enumerate(masks):

                mask = masks[i]
                mask = mask.astype('uint8')

                # Add points to image 
                vis_points = cv2_img.copy()
                for i, [x, y] in enumerate(points):
                    if labels[i] == 1:
                        cv2.circle(vis_points, (x, y), 3, (0, 0, 255) if labels[i] == 1 else (255, 0,0), 3) 
            

                # Visualize mask 
                img_with_mask = show_mask(mask, cv2_img.copy(), random_color=False, opacity=0.6)

                #Highlight Combbine Features
                
                vis_features = (similarity_map[:, :, n].cpu().numpy()*255).astype('uint8')
                vis_features = cv2.applyColorMap(vis_features, cv2.COLORMAP_JET)
                vis_features = cv2_img.copy() * 0.4 + vis_features * 0.6
                
                
                # Save image
                all_vis_img = np.hstack((cv2_img.copy(), vis_features, vis_points, img_with_mask))
                cv2.imwrite("{}_{}_mask_{}.jpg".format(os.path.basename(args.img), target_texts[n], i), all_vis_img)
                

                print('SAM guided by points from CLIP Surgery:', target_texts[n])
               

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=str, default='0', help="GPU to use")
    parser.add_argument("--img", type=str, default="/home/khiem/segment-anything/demo_imgs/multi_obj.jpg", help="image")
    args = parser.parse_args()
    main(args)