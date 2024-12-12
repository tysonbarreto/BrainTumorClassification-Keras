
import cv2, os, sys
from keras._tf_keras.keras.preprocessing.image import load_img, save_img
from typing import Sequence, Dict, Any
import imutils
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class Preprocess_Images:

    @staticmethod
    def crop_image(image:os.PathLike, plot=False):
        try:
            ex_image= cv2.imread(image)
            img_gray = cv2.cvtColor(src=ex_image, code=cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(src=img_gray, ksize=(5, 5), sigmaX=0, sigmaY=0)
            img_thresh = cv2.threshold(src=img_blur, thresh=45, maxval=255, type=cv2.THRESH_BINARY)[1]
            img_thresh = cv2.erode(src=img_thresh,kernel=None, iterations=2)
            img_thresh = cv2.dilate(src=img_thresh, kernel=None, iterations=2)
            contours = cv2.findContours(image=img_thresh.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            contours = imutils.grab_contours(contours)
            c = max(contours, key = cv2.contourArea)

            extLeft = tuple(c[c[:, :, 0].argmin()])[0]
            extRight = tuple(c[c[:, :, 0].argmax()])[0]
            extTop = tuple(c[c[:, :, 1].argmin()])[0]
            extBottom = tuple(c[c[:, :, 1].argmax()])[0]

            new_img = ex_image[extTop[1]: extBottom[1], extLeft[0]:extRight[0]]

            if plot:
                plt.figure(figsize = (12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(ex_image)
                plt.title("Original Image")
                plt.subplot(1, 2, 2)
                plt.imshow(new_img)
                plt.title("Cropped Image")
                plt.show()
            return new_img
        except Exception:
            print(Exception(f"Error occured with cropping file {image}"))

    def save_cropped_images(self, classes:Sequence[str],filepath_dict:Dict[str,Sequence[os.PathLike]], save_path_folder:os.PathLike)->None:

        with tqdm(classes, leave=True) as class_:
            for c in class_:
                class_.set_description(desc=c)
                for i,image_path in tqdm(enumerate(filepath_dict[c]), desc=f"{c}-{len(filepath_dict[c])} images",total=len(filepath_dict[c]), leave=True):
                    image_path=os.path.normpath(image_path)
                    cropped_image = self.crop_image(image_path, plot=False)

                    if cropped_image is not None:
                        resized_img = cv2.resize(src=cropped_image,dsize=(240,240))
                        save_path = os.path.join(save_path_folder, c, f"{i}.jpg")
                        if not os.path.exists(save_path):
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        cv2.imwrite(filename=save_path, img=resized_img)

if __name__=="__main__":
    __all__=["Preprocess_Images"]

