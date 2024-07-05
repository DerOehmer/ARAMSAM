import cv2
import numpy as np
from .sam_mask_creation import AddMasks, MaskData
import time
class ShowMe():
    def __init__(self, img, factor = 1.0):
        self.imgls = img
        self.factor = factor
        self.__call__()
        
    def __call__(self):
        for i, img in enumerate(self.imgls):
            cv2.imshow(f"img{i}",self.resize(img))
        key = cv2.waitKey(0) & 0xFF
        if key == ord("x"):
            quit()
        elif key == ord("c"):
            cv2.destroyAllWindows()
            pass

    def resize(self, img):
        shape0 = int(img.shape[0] * self.factor)
        shape1 = int(img.shape[1] * self.factor)
        imgr = cv2.resize(img, (shape1, shape0))

        return imgr

class ImgCheckup():
    def __init__(self, imgobj):
      
        self.imgobj = imgobj
        self.img = imgobj.img
        self.cnt_col = imgobj.annotated_img
        
        
        
    def show(self):
        
        while True:
            print("All Kernels detected correctly?")
            print("Press (y) for saving masks as done.")
            print("Press (r) to repeat the current image.")
            print("Press (p) to start drawing polygons.")
            print("Press (s) to start interactive prompting with SAM.")
            imgls = [self.imgobj.img, self.imgobj.mask_collection, self.imgobj.annotated_img]
            
            for i, img in enumerate(imgls):
                if i == 0:
                    continue
                elif i==1:
                    showarray = imgls[i-1] 
                if len(img.shape) < 3:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                showarray = np.hstack((showarray, img))                

            imshow_(f"img{i}",showarray)
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord("x"):
                raise KeyboardInterrupt()
            elif key == ord("y"):
                cv2.destroyAllWindows()
                return False, self.imgobj
            elif key == ord("r"):
                print("Current img will be repeated")
                cv2.destroyAllWindows()
                return True, self.imgobj
            elif key == ord("p"):
                print("Starting polygon drawing")
                cv2.destroyAllWindows()
                addmasks = AddMasks(self.imgobj, procedure="polygon_drawing")
                self.imgobj = addmasks()
            elif key == ord("s"):
                if self.imgobj.sam is None:
                    print("SAM model not loaded, cannot start SAM prompting")
                    continue
                print("Starting interactive SAM prompting")
                cv2.destroyAllWindows()
                addmasks = AddMasks(self.imgobj, procedure="sam_prompting")
                self.imgobj = addmasks()
            else:
                print("Unrecognised character")


    
class SelectMask():
    def __init__(self, imgls, orig_img, collections, mask: MaskData, mask_lsts, skip_tracked = False):
        self.imgls = imgls
        self.orig_img = orig_img
        
        self.mask_col, self.cnt_col = collections
        self.mask = mask.mask
        self.maskobj = mask
        self.good_mask_lst, self.dec_lst = mask_lsts
        self.skip_tracked = skip_tracked
        
        
    def show(self):
        early_finish = False
        step_back = False
        while True:
            showarray = np.hstack((self.draw_future_location(self.orig_img),self.draw_future_location(self.mask_col, thickness=2)))
            showarray = np.hstack((showarray,cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)))
            for img in self.imgls:
                showarray = np.hstack((showarray,img))
                
                
            if not self.skip_tracked:
                imshow_('array',showarray)
            
                key = cv2.waitKey(0) & 0xFF
            else:
                key = ord("n")

            if key == ord("x"):
                cv2.destroyAllWindows()
                raise KeyboardInterrupt()

            elif key == ord("n"):
                self.append_mask()
                self.append_decision(True)
                #self.cnt_col = self.draw_future_location(self.cnt_col, fill=True)
                self.cnt_col = self.draw_cnt_coll()
                #cv2.destroyAllWindows()
                break

            elif key == ord("m"):
                self.append_decision(False)
                #cv2.destroyAllWindows()
                break

            elif key == ord("b"):
                step_back = True
                #cv2.destroyAllWindows()
                break

            elif key == ord("f"):
                print("All releveant masks have been selected, moving on to next image")
                cv2.destroyAllWindows()
                early_finish = True
                break

            else:
                print("Unrecognised character")

            
        collections = self.mask_col, self.cnt_col
        mask_lsts = self.good_mask_lst, self.dec_lst
        return collections,  mask_lsts, early_finish, step_back
    
    def append_mask(self):
        self.good_mask_lst.append(self.maskobj)

        mask_coll_bin = np.any(self.mask_col != [0, 0, 0], axis=-1).astype(np.uint8) * 255
        overlap = cv2.bitwise_and(self.mask, mask_coll_bin)
        mask_coll_bin = np.where(self.mask==255, 255, mask_coll_bin)
        self.mask_col[np.where(self.mask==255)] = [255,255,255]
        self.mask_col[np.where(overlap==255)] = [0,0,255]

    def append_decision(self, res):
        self.dec_lst.append(res)


    def draw_future_location(self, dest, fill=False, thickness=1):
        cnts, _ = cv2.findContours(self.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
       
        marked0 = dest.copy()

        if fill:
            thickness = -1
            b, g, r = np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)
            marked = cv2.drawContours(marked0, cnts, -1, (b,g,r), thickness, lineType=cv2.LINE_8)
        else:
            b, g, r = 0,200,0
            marked = cv2.drawContours(marked0, cnts, -1, (b,g,r), thickness, lineType=cv2.LINE_8)
        return marked
    
    def draw_cnt_coll(self):
        thickness = -1
        marked0 = self.orig_img.copy()
        for m in self.good_mask_lst:
            cnts, _ = cv2.findContours(m.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            b, g, r = np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)
            marked = cv2.drawContours(marked0, cnts, -1, (b,g,r), thickness, lineType=cv2.LINE_8)

        return marked
    
def imshow_ (windowname, img, factor=1.0, rotation = True):
    
    if factor == 1.0 and img.shape[0] > 1920:
        factor =  1920 / img.shape[0]
    elif factor == 1.0 and img.shape[1] > 1920:
        factor =  1920 / img.shape[1]
    shape0 = int(img.shape[0] * factor)
    shape1 = int(img.shape[1] * factor)
    imgshow = cv2.resize(img, (shape1, shape0))
    if rotation:
        imgshow = cv2.rotate(imgshow, cv2.ROTATE_90_CLOCKWISE)

    cv2.imshow(windowname,imgshow)