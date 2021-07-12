import numpy as np
import pydicom
import matplotlib.pyplot as plt
import cv2 as cv
import albumentations as A
from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage.filters import unsharp_mask
from skimage import exposure

TARGET_SIZE = 600

class Kaggle_agu:
    ### yixuetu to huidutu
    def dicom2array(self, path, voi_lut=True, fix_monochrome=True):
        """
        Transform a dicom file to an array.

        - path : path of the dicom file
        - voi_lut : Apply VOI LUT transformation
        - fix_monochrome : Indicate if we fix the pixel value for specific files.

        VOI LUT (Value of Interest - Look Up Table) : The idea is to have a larger representation of the data.
        Since, dicom files have larger pixel display range than usuall pictures. The idea is to keep a larger representation in order ot better see the subtle differences.

        Fix Monochrome : Some images have MONOCHROME1 interpretation. Which means that higher pixel values corresponding to the dark instead of the white.
        """
        dicom = pydicom.read_file(path)

        # Apply the VOI LUT
        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array

        # Fix the representation
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

        data = data - np.min(data)
        data = data / np.max(data)



        data = data * 255

        data = data.astype(np.uint8)

        return data

    ##ke shi hua
    def img_vizualisation(self, imgs, nb_samples=5):

        fig, axes = plt.subplots(nrows=nb_samples // 5, ncols=min(5, nb_samples),
                                 figsize=(min(5, nb_samples) * 4, 4 * (nb_samples // 5)))
        i = 0
        for img in imgs:
            axes[i // 5, i % 5].imshow(np.array(img), cmap=plt.cm.gray, aspect='auto')
            axes[i // 5, i % 5].axis('off')
            i += 1
        fig.show()

    ## zi shi ying zhi fang tu jun heng hua
    def CLAHE(self, img):
        clahe = cv.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)
        return img_clahe

    ##yi xue duibidu zeng qiang
    def contrast_enhance(self, img):
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))  # MORPH_ELLIPSE

        tophat_img = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
        bothat_img = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)  # Black --> Bottom

        output = img + tophat_img - bothat_img
        return output

    ## jie he 3 ge tong dao
    def combin_chanel(self, chanel1, chanel2, chanel3):
        output = np.dstack((chanel1, chanel2, chanel3))
        return output


    ##fei rui hua meng ban & zhi fang tu jun zhi hua
    def unsharp_image(self, img):
        # Tweak the radius and amount for more/less sharpening
        unsharp_image = unsharp_mask(img, radius=5, amount=2)
        equalized_image = exposure.equalize_hist(unsharp_image)
        return equalized_image

    ### sui ji zeng qiang
    def random_img_aru(self, img):
        ### all type
        all_albumentations = [A.RandomSunFlare(p=1), A.RandomFog(p=1), A.RandomBrightness(p=1),
                          A.RandomCrop(p=1, height=128, width=128), A.Rotate(p=1, limit=90),
                          A.RGBShift(p=1), A.RandomSnow(p=1),
                          A.HorizontalFlip(p=1), A.VerticalFlip(p=1), A.RandomContrast(limit=0.5, p=1),
                          A.HueSaturationValue(p=1, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]
        ## choosed
        use_albumentations = [A.RandomBrightness(p=1), A.HorizontalFlip(p=1), A.RandomContrast(limit=0.5, p=1), A.Rotate(p=1, limit=90)]
        argu_img1 = use_albumentations[0](img = img)['image']
        argu_img2 = use_albumentations[1](img=img)['image']
        argu_img3 = use_albumentations[2](img=img)['image']
        argu_img4 = use_albumentations[3](img=img)['image']
        return argu_img1, argu_img2, argu_img3, argu_img4
