import cv2
import numpy as np
import os
import glob
from natsort import natsorted
import pandas as pd
from collections import defaultdict
from time import time
import shutil
import argparse


from src.sam_mask_inspection import SelectMask, ShowMe, ImgCheckup
from src.sam_mask_creation import MaskData, ImageData, ImageAligner, update_collections   


    

def basic_folder_structure( output_path, cob_name):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    cob_folder = f"{output_path}/{cob_name}" 
    if not os.path.exists(cob_folder):
        os.mkdir(cob_folder)

    done_folder = f"{cob_folder}/done" 
    
    
    if not os.path.exists(done_folder):
        os.mkdir(done_folder)

    return done_folder

def sort_masks(masks, righttoleft=True):
    sorted_masks = []
    ymins = []
    xmins = []
    for mobj in masks:
        m = mobj.mask
        ycords, xcords = np.where(m == 255)
        ymin, xmin = np.amin(ycords), np.amin(xcords)

        sorted_masks.append(m)
        ymins.append(ymin)
        xmins.append(xmin)

    cord_dict = {"ymins" : ymins,
                 "xmins" : xmins}
    
    cord_df = pd.DataFrame(cord_dict)
    cord_df = cord_df.sort_values(by=['ymins', 'xmins'], ascending=righttoleft)
    order = cord_df.index.to_list()
    #sorted_masks = np.array(sorted_masks, dtype=np.uint8)[order]
    sorted_masks = [masks[i] for i in order]

    return sorted_masks

def get_prev_imgobj(prev_donep):
    img_name = os.path.basename(prev_donep)
    imgp = f"{prev_donep}/{img_name}_img.png"
    if not os.path.exists(imgp):
        raise FileNotFoundError("Previous image not found", imgp)
    img = cv2.imread(imgp)

    maskps = glob.glob(f"{prev_donep}/masks/*")
    masks = read_masks(maskps, origin="sam_tracked")
    return ImageData(img, None, None, masks, None)

def mask_prep(imgobj_raw : ImageData, 
              out_folders : str,  
              img_name: str, 
              imgalign : ImageAligner, 
              log_dict : defaultdict, 
              max_overlap_ratio : float=.4 ):
    
    img_done_folder,  mask_done_folder, prev_donep = out_folders
    img = imgobj_raw.img
    h, w, _ = img.shape

    repeat = True
    step_back = False
    sam_suggestions = imgobj_raw.annotated_img
    masks = imgobj_raw.masks 
    pre_tracked_masks = []   
    
    if sam_suggestions is None and masks == []:
        sam = SamInference(sam_checkpoint=r"C:\Users\geink81\Desktop\pythonstuff\EarScanLabeling\BestGD.pth")
        kernel_masks, sam_suggestions = sam.amg(img)
    else:
        kernel_masks_raw = sort_masks(masks)

    

    if len(imgalign.images) >= 1:
        imgalign.add_image(imgobj_raw)
        print("Matching and aligning images...")
        pre_tracked_masks = imgalign.match_and_align()
        pre_tracked_masks = sort_masks(pre_tracked_masks, righttoleft=False)
        kernel_masks = pre_tracked_masks + kernel_masks_raw
    elif prev_donep is not None:
        prev_imgobj = get_prev_imgobj(prev_donep)
        imgalign.add_image(prev_imgobj)
        imgalign.add_image(imgobj_raw)
        print("Matching and aligning images...")
        pre_tracked_masks = imgalign.match_and_align()
        pre_tracked_masks = sort_masks(pre_tracked_masks, righttoleft=False)
        kernel_masks = pre_tracked_masks + kernel_masks_raw
    else:
        kernel_masks = kernel_masks_raw

    while repeat:
        maski = 0
        good_masks = []
        mask_dec = []
        mask_collection = np.zeros((h, w, 3), dtype=np.uint8)
        cnt_collection = img.copy()
        print("Press (n) for keeping mask.")
        print("Press (m) for discarding mask.")
        print("Press (b) to go back to previous mask.")
        print("Press (f) if all relevant masks have been selected already.")
        print("Press (x) to cancel script.")
        
        while maski < len(kernel_masks):
            kernelmask_obj = kernel_masks[maski]
            kernelmask = kernelmask_obj.mask
            maskorigin = kernelmask_obj.origin

            imgresult = cv2.bitwise_and(img, img ,mask=kernelmask)
           
            show_lst = [sam_suggestions, imgresult]

            """mask_size = np.count_nonzero(kernelmask)
            mask_overlap = cv2.bitwise_and(mask_collection, mask_collection, mask=kernelmask)
            mask_overlap_size = np.count_nonzero(mask_overlap)
            overlap_ratio = mask_overlap_size  / mask_size"""
            mask_coll_bin = np.any(mask_collection != [0, 0, 0], axis=-1).astype(np.uint8) * 255
            mask_overlap = cv2.bitwise_and(mask_coll_bin,kernelmask)
            mask_size = np.count_nonzero(kernelmask)
            mask_overlap_size = np.count_nonzero(mask_overlap)
            overlap_ratio = mask_overlap_size  / mask_size

            if overlap_ratio > max_overlap_ratio and not step_back:
                maski += 1
                mask_dec.append(False)
                continue

            if maskorigin == "sam_tracking" and not step_back and len(mask_dec) == len(good_masks):
                skip_tracked = True
            else:
                skip_tracked = False

            collections = mask_collection, cnt_collection
            mask_lsts = good_masks, mask_dec
            step_back = False
            
            (mask_collection, cnt_collection), (good_masks, mask_dec), early_finish, step_back  = SelectMask(
                                                                                     show_lst, 
                                                                                     img, 
                                                                                     collections, 
                                                                                     kernelmask_obj, 
                                                                                     mask_lsts,
                                                                                     skip_tracked
                                                                                     ).show()
            if step_back:
                if maski > 0:
                    maski -= 1
                    if mask_dec[-1]:
                        good_masks.pop()
                        mask_collection, cnt_collection = update_collections(good_masks, img)
                    mask_dec.pop()   
            else:
                maski += 1

            if early_finish:
                break

        
        cv2.destroyAllWindows()

        imgobj_new = ImageData(img, 
                           cnt_collection,
                           mask_collection,
                           #[MaskData(mask, "sam_selecting") for mask in good_masks],
                           good_masks,
                           samseg)
        
        repeat, imgobj_new = ImgCheckup(imgobj_new).show()

        

        
    print("--------------------------------")
    imgalign.add_image(imgobj_new)

    cv2.imwrite(f"{img_done_folder}/{img_name}_img.png", img)
    cv2.imwrite(f"{img_done_folder}/{img_name}_SelectedInstances.png", imgobj_new.annotated_img)
    suggest_count, track_count, interact_count, draw_count = 0,0,0,0
    for m in imgobj_new.masks:
        mask = m.mask
        orig = m.origin
        if orig == "sam_selecting":
            suggest_count += 1
            i = suggest_count
        elif orig == "sam_tracking":
            track_count += 1
            i = track_count
        elif orig == "sam_prompting":
            interact_count += 1
            i = interact_count
        elif orig == "polygon_drawing":
            draw_count += 1
            i = draw_count
        else:
            print("Maks origin: ", orig)
            raise ValueError("Unknown mask origin")
        
        mask_path = f"{mask_done_folder}/{img_name}_{orig}_mask{i}.png"
        cv2.imwrite(mask_path, mask)

    log_dict["sam_proposed"].append(len(kernel_masks_raw))
    log_dict["sam_tracked"].append(len(pre_tracked_masks))
    log_dict["sam_selecting"].append(suggest_count)
    log_dict["sam_tracking"].append(track_count)
    log_dict["sam_prompting"].append(interact_count)
    log_dict["polygon_drawing"].append(draw_count)
    return log_dict


def make_folders(done, addition):
    sub_done_folder = f"{done}/{addition}"
  
   
    if not os.path.exists(sub_done_folder):
        os.mkdir(sub_done_folder)

    return sub_done_folder

def read_masks(mask_paths, origin="sam_selecting"):
    masks = []
    for m_path in mask_paths:
        mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        if len(np.unique(mask)) > 2:
            _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY) #Make sure mask is binary
        masks.append(MaskData(mask, origin))
    return masks

def sort_paths(eardirp):
    ear_dirs = glob.glob(eardirp + "/*"	)
    # Sort the paths: directories first, then .zip files
    ear_dirs = sorted(ear_dirs, key=lambda x: x.endswith('.zip'))
    return ear_dirs

def count_subdirectories(directory):
    try:
        return sum([len(dirs) for _, dirs, _ in os.walk(directory)])
    except Exception as e:
        return 0

def sort_paths(eardirp):
    paths = glob.glob(eardirp + "/*"	)
    directories = [path for path in paths if not path.endswith('.zip')]
    zip_files = [path for path in paths if path.endswith('.zip')]
    
    # Sort directories based on the number of subdirectories they contain in descending order
    directories.sort(key=count_subdirectories, reverse=True)
    
    # Combine directories and zip files
    sorted_paths = directories + zip_files
    return sorted_paths
   
        
if __name__ == "__main__":
    SAMP = "BestGD.pth"
    EARDIRP = os.path.join("jessiraw")

    parser = argparse.ArgumentParser(description='Arguments for semi-auto labeling script')

    parser.add_argument('--path', type=str, default=EARDIRP, help='Ear directory path')
    args = parser.parse_args()

    PREPARED_MASKS = True
    DONECHECK = False
    IMPORTSAM = True #If true, sam can be used interactively. Requires torch and segment anything model.

    EARDIRP = args.path	

    ear_dirs = sort_paths(EARDIRP)
   
    if IMPORTSAM:
        from src.run_sam import SamInference 
        samseg = SamInference(SAMP,"vit_b") 

    for ear_path in ear_dirs:
        prev_donep = None

        if ear_path.endswith("_done"):
            continue

        if ear_path.endswith(".zip"):
            ear_path_new = os.path.splitext(ear_path)[0]
            if not os.path.exists(ear_path_new):
                print("Unzipping", ear_path)
                shutil.unpack_archive(ear_path, ear_path_new, 'zip')
                print("Unzipping done!")
            ear_path = ear_path_new

          

        output_path = ear_path + "_done"

        if not os.path.exists(ear_path):
            raise FileNotFoundError("Input path not found")
        
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        cob_name = os.path.basename(ear_path)
        #done_folder = basic_folder_structure(output_path, cob_name)
        
        path_list = natsorted(glob.glob(ear_path+"/*"))

        imagealign = ImageAligner()

    
        for img_path in path_list:
            
            print("#################################")
            print(img_path)
            #if not "3136" in img_path:
                #continue
            img_name = os.path.basename(img_path)
            
            
            img_done_folder= make_folders(output_path,img_name) 
        

            mask_done_folder = make_folders(img_done_folder,"masks")
            output_folders = img_done_folder, mask_done_folder, prev_donep
            
            done_folder_ps = glob.glob(img_done_folder+"/*")
            if any(file.endswith(('.csv', '.json')) for file in done_folder_ps):
                prev_donep = img_done_folder
                print("Skipping ", img_done_folder)
                continue 
            log_dict = defaultdict(lambda: [])
            log_dict["Image"].append(img_name)
            img = cv2.imread(img_path + "/img.jpg")
            if PREPARED_MASKS:
                sam_suggestions = cv2.imread(img_path + "/annotations.jpg")
                mask_paths = glob.glob(img_path +"/masks/*")
                masks = read_masks(mask_paths)
            elif DONECHECK:
                sam_suggestions = cv2.imread(img_path + "/annotations.jpg")
                mask_paths = glob.glob(img_path +"/masks/*")
                masks = read_masks(mask_paths)
            else:
                sam_suggestions = None
                mask_paths = []
            if IMPORTSAM:
                samseg.image_embedding(img)
            else:
                samseg = None

            imgob_jraw = ImageData(img, sam_suggestions, None, masks, samseg)
            annotstart = time()
            #imagealign.add_image(img)
            log_dict = mask_prep(imgob_jraw, output_folders, img_name, imagealign, log_dict)
            annotend = time()
            log_dict["Annotation_time"].append(annotend - annotstart)
            
            logdf = pd.DataFrame(log_dict)
            logdf.to_csv( f"{img_done_folder}/{img_name}_log.csv",index=False)

            if IMPORTSAM: samseg.reset_img()
        
