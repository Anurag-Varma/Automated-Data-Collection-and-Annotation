import numpy as np
import cv2
from PIL import Image, ImageTk

class MaskGenerator:

    def __init__(self, predictor) -> None:
        self.predictor = predictor
        self.coord_available = True

    def setCoordAvailable(self,yes):
        self.coord_available = yes
    
    def setMask(self, mask) -> None:
        self.mask  = mask
        self.setCoordAvailable(False)

    def setCoordinates(self, x,y) -> None:
        self.x = x
        self.y = y
        labels = [1]
        masks, scores, logits = self.predictor.predict(point_labels=labels, point_coords=np.array([[x,y]]), multimask_output=True)
        self.mask  = masks[np.argmax(scores)].astype("uint8")

    def predictByArray(self, points) -> None:
        print(points)
        labels = np.array([1]*len(points))
        print(labels)
        masks, scores, logits = self.predictor.predict(point_labels=labels, point_coords=points, multimask_output=True)
        self.mask  = masks[np.argmax(scores)].astype("uint8")

    def show_mask(self, mask, img, random_color, opacity):
        if random_color:
            color = np.concatenate([np.random.random(3)*255], axis=0)
        else:
            color = np.array([30, 144, 255])
        h, w = mask.shape[-2:]
        color_seg = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        fg_mask = mask != False
    
        img[fg_mask] = color_seg[fg_mask] * opacity
    
        return img
    
    def cv2_to_imageTK(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        imagePIL = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image = imagePIL)
        return imgtk
    
    def printMask(self,img,label)->None:
        img_with_mask = self.show_mask(self.mask, img.copy(), False, 0.6)
        if self.coord_available:
            cv2.circle(img_with_mask, (self.x, self.y), 3, (255, 0, 0), 3) 
        imgtk = self.cv2_to_imageTK(img_with_mask)
        label.imgtk = imgtk
        label.configure(image = imgtk)


 