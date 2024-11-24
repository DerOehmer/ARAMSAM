import cv2
import numpy as np
import glob
import os


def highlight_masks_on_img(img_p: str, mask_ps: list):
    img = cv2.imread(img_p)
    for mask_p in mask_ps:
        mask = cv2.imread(mask_p)
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(
            gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(
                img, (x - 10, y - 10), (x + w + 10, y + h + 10), (74, 163, 2), 3
            )
    return img


def get_img_and_masks_paths(img_dir: str, n_masks: int = 2):
    imgps = []
    maskps = []
    for ear_view in glob.glob(img_dir + "/*"):
        mask_ps = glob.glob(os.path.join(ear_view, "masks") + "/*")
        rand_mask_idcs = np.random.choice(len(mask_ps), n_masks, replace=False)
        rand_mask_ps = [mask_ps[i] for i in rand_mask_idcs]
        imgps.append(ear_view + "/img.jpg")
        maskps.append(rand_mask_ps)

    return imgps, maskps


if __name__ == "__main__":
    ear_img_dir = "ExperimentData/BackboneExperimentData/MaizeEar"
    img_dest = "ExperimentData/IndicatedPolygonPositionImages"
    if not os.path.exists(img_dest):
        os.makedirs(img_dest)
    ear_img_paths, mask_paths = get_img_and_masks_paths(ear_img_dir)
    print(ear_img_paths, mask_paths)
    for ear_img_p, mask_ps in zip(ear_img_paths, mask_paths):
        img_name = os.path.basename(os.path.dirname(ear_img_p))
        ind_img_name = os.path.join(img_dest, img_name) + "_indicated.jpg"
        indicated_img = highlight_masks_on_img(ear_img_p, mask_ps)
        cv2.imwrite(ind_img_name, indicated_img)
