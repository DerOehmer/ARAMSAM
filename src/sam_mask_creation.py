import cv2
import numpy as np
import os
from collections import defaultdict
from dataclasses import dataclass, is_dataclass
from scipy.spatial import KDTree

@dataclass
class MaskData:
    mask: np.ndarray
    origin: str

@dataclass
class ImageData:
    img: np.ndarray
    annotated_img: np.ndarray
    mask_collection: np.ndarray
    masks: list
    sam: None



def update_collections(good_mask_lst, rgb_img):
    y, x, _ = rgb_img.shape
    mask_coll = np.zeros((y,x,3), dtype=np.uint8)
    mask_coll_bin = np.zeros((y,x), dtype=np.uint8)
    cnt_coll = rgb_img.copy()
 
    thickness = -1

    for m in good_mask_lst: 
        if is_dataclass(m):
            m=m.mask
        overlap = cv2.bitwise_and(m, mask_coll_bin)
        mask_coll_bin = np.where(m==255, 255, mask_coll_bin)
        mask_coll[np.where(m==255)] = [255,255,255]
        mask_coll[np.where(overlap==255)] = [0,0,255]

        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        b, g, r = np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)
        cnt_coll = cv2.drawContours(cnt_coll, cnts, -1, (b,g,r), thickness, lineType=cv2.LINE_8)

    return(mask_coll, cnt_coll)

class ImageAligner:
    def __init__(self):
        self.images = []
        self.masks = []
        self.h_matrix = None

    def add_image(self, imgobj):
        imgbgr = imgobj.img
        img_gray = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2GRAY) 

        # Add the image to the list
        self.images.append(img_gray)
        self.masks.append(imgobj.masks)
        if len(self.images) > 2:
            # Keep only the two most recent images
            self.images.pop(0)
            self.masks.pop(0)
        
    def match_and_align(self):
        img_gray1, img_gray2 = self.images

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img_gray1, None)
        kp2, des2 = orb.detectAndCompute(img_gray2, None)

        # Match descriptors.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Sort matches by distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw top matches.
        #img_matches = cv2.drawMatches(img_gray1, kp1, img_gray2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Estimate homography.
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        self.h_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp image using homography.
        height, width = img_gray2.shape

        tracked_masks = self.warp_masks(self.masks[0], width, height)
        selected_masks, centerpts = self.select_masks_with_highest_iou(tracked_masks, self.masks[1])

        """annotimg = cv2.cvtColor(img_gray2.copy(), cv2.COLOR_GRAY2BGR)

        
        for m in self.masks[1]:
            mask = m.mask
            annotimg = self.get_mask_cnts(annotimg, mask, col=(10,10,10))
        for m in tracked_masks:
            annotimg = self.get_mask_cnts(annotimg, m, col=(100,100,100))
        for m in selected_masks:
            #mask = cv2.warpPerspective(mask, self.h_matrix, (width, height))
            annotimg = self.get_mask_cnts(annotimg, m, col=(255,0,0))
        for pt in centerpts:
            y,x = pt
            annotimg = cv2.circle(annotimg, (x,y), 2, (0, 255, 0), -1)

        # Save and display the results.
        ShowMe([annotimg], factor=1)"""
        return [MaskData(mask, "sam_tracking") for mask in selected_masks]
    
    def warp_masks(self, masks,w, h):
        return [cv2.warpPerspective(warp_m.mask, self.h_matrix, (w, h),flags=cv2.INTER_NEAREST) for warp_m in masks]
    
    def get_mask_cnts(self, annnotimg, mask, thickness = 2, col=None):
        cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if col is None:
            b, g, r = np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)
        else:
            b, g, r = col
        annnotimg = cv2.drawContours(annnotimg, cnts, -1, (b,g,r), thickness, lineType=cv2.LINE_8)
        return annnotimg

    def compute_iou(self, mask1, mask2):
        # Compute intersection
        intersection = np.logical_and(mask1, mask2).sum()
        # Compute union
        union = np.logical_or(mask1, mask2).sum()
        # Compute IoU
        iou = intersection / union if union != 0 else 0
        return iou
    
    def compute_centroid(self, mask):
        indices = np.argwhere(mask)
        if len(indices) == 0:
            return None
        centroid = np.mean(indices, axis=0, dtype=np.int32)
        return centroid


    def select_masks_with_highest_iou(self, tracked_masks, new_masks, threshold=0.7, k=3):
        # Compute centroids for all masks in tracked masks
        centroids_tr = []
        tracked_masks_roi = []
        for maskt in tracked_masks:
            centerpt = self.compute_centroid(maskt)
            if centerpt is not None:
                centroids_tr.append(centerpt)
                tracked_masks_roi.append(maskt)
        
        # Build a KDTree with the centroids of tracked masks
        tracked_tree = KDTree(centroids_tr)
        
        selected_masks = []
        for mn in new_masks:
            mask_n = mn.mask
            centroid_new = self.compute_centroid(mask_n)
           
            # Find the k nearest masks in tracked masks to the mask in new masks
            distances, indices = tracked_tree.query(centroid_new, k=k)
            best_mask = None
            max_iou = 0
            for idx, dist in zip(indices, distances):
                
                mask_t = tracked_masks_roi[idx]
                
                iou = self.compute_iou(mask_t, mask_n)
                #print("IoU: ", iou)
                #print("Distance: ", dist)
                #ShowMe([mask_t, mask_n])
                if iou > max_iou:
                    max_iou = iou
                    best_mask = mask_n
            
            if max_iou > threshold:
                selected_masks.append(best_mask)
                #ShowMe([best_mask], .7)
        

        highest_iou_masks = np.array([mask for mask in selected_masks], dtype=np.uint8)

        return highest_iou_masks, centroids_tr


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
            raise KeyboardInterrupt()
        elif key == ord("c"):
            cv2.destroyAllWindows()
            pass

    def resize(self, img):
        shape0 = int(img.shape[0] * self.factor)
        shape1 = int(img.shape[1] * self.factor)
        imgr = cv2.resize(img, (shape1, shape0))

        return cv2.rotate(imgr, cv2.ROTATE_90_CLOCKWISE)

class PromptRegistration:
    def __init__(self, img):
        # Initialize variables
        self.img = img
        self.ptimg = None
        self.rand_col = self.random_color()
        self.done = False
        self.points = []
        self.current = (0, 0)
        self.prev_current = (0, 0)

        # Set up the named window and mouse callback
        #cv2.namedWindow("image")
        #cv2.setMouseCallback("Preview", self.on_mouse)
    
    def random_color(self) -> list[int]:
        """Get a random RGB color."""
        return np.random.randint(0, 255, 3).tolist()

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event
        if self.done:  # Nothing more to do
           return
        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            #print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.ptimg = cv2.circle(self.ptimg, (x, y), 5, (0, 200, 0), -1)
            self.points.append([x, y])
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points = self.points[:-1]
            self.reset_pt_annots()

    def reset_pt_annots(self):
        self.ptimg = self.img.copy()
        for pt in self.points:
            self.ptimg = cv2.circle(self.ptimg, pt, 5, (0, 200, 0), -1)       

    def draw_polygon(self):

        cv2.namedWindow("Preview")
        cv2.setMouseCallback("Preview", self.on_mouse)
        print("Left click to select next vertex.")
        print("Right click to delete last vertex.")
        print("Press (n) to close Polygon and preview mask.")
        print("Press (m) to go back to region selection.")
        print("-----------------------------------")
        self.ptimg = self.img.copy()
        while not self.done:
            
            if len(self.points) > 1:
                #if self.current != self.prev_current:
                    #self.ptimg = self.temp.copy()
                self.ptimg = cv2.polylines(self.ptimg, [np.array(self.points)], False, (255, 0, 0), 1)
                #self.ptimg = cv2.line(self.ptimg, (self.points[-1][0], self.points[-1][1]), self.current, (0, 0, 255))

            cv2.imshow("Preview", self.ptimg)
            prevkey= cv2.waitKey(50) & 0xFF
            
            if prevkey== ord('n'):
                self.done = True
            if prevkey== ord('m'):
                cv2.destroyWindow("Preview")
                return []
            elif prevkey== ord("x"):
                raise KeyboardInterrupt()
            
        cv2.destroyWindow("Preview")
        # Drawing the final filled polygon
        maskimg = self.ptimg.copy()
        if len(self.points) > 0:
            cv2.fillPoly(maskimg, np.array([self.points]), self.rand_col)
        
        print("press (n) to safe mask.")
        print("press (m) to delete mask.")
        print("-----------------------------------")
        
        while True:
            cv2.imshow("Mask view", maskimg)
            maskkey = cv2.waitKey(0) & 0xFF
        
            if maskkey == ord('n'): #safe
                cv2.destroyWindow("Mask view")
                return self.points
            elif maskkey == ord('m'): #delete
                cv2.destroyWindow("Mask view")
                return []
            elif maskkey == ord("x"):
                cv2.destroyWindow("Mask view")
                raise KeyboardInterrupt()
    
    def draw_bbox(self):
        cv2.namedWindow("Preview")
        cv2.setMouseCallback("Preview", self.on_mouse)
        self.ptimg = self.img.copy()
        print("press (n) to safe mask.")
        print("press (m) to delete mask.")
        print("-----------------------------------")
        while True:
            
            if len(self.points) == 2:
                self.ptimg = cv2.rectangle(self.ptimg, self.points[0], self.points[1], (255, 0, 0), 1)
              
            elif len(self.points) == 3:
                self.points = [self.points[0], self.points[-1]]
                self.reset_pt_annots()
                continue


            cv2.imshow("Preview", self.ptimg)
            prevkey= cv2.waitKey(50) & 0xFF
            
            if prevkey== ord('n') and len(self.points) == 2:
                cv2.destroyWindow("Preview")
                self.points = self.ensure_correct_pt_order(self.points[0], self.points[1])
                return self.points
            
            elif prevkey == ord('m'): #delete
                cv2.destroyWindow("Preview")
                return []
            
            elif prevkey== ord("x"):
                cv2.destroyWindow("Preview")
                raise KeyboardInterrupt()
            
    def ensure_correct_pt_order(self, pt1, pt2):
        # Unpack the points
        x1, y1 = pt1
        x2, y2 = pt2

        # Check if pt1 is actually the bottom-right or pt2 is actually the top-left
        if x1 > x2 or y1 > y2:
            # Swap points if pt1 is not the top-left or pt2 is not the bottom-right
            return [pt2, pt1]
        else:
            # Return points as they are if already in correct order
            return [pt1, pt2]
        

    def get_color(self):
        return self.rand_col         
    
    def get_points(self):
        return self.points

class AddMasks():
    #def __init__ (self, img, mask_path=None, log=defaultdict(lambda: []), procedure="polygon_drawing"):
    def __init__ (self, imgobj, procedure="polygon_drawing"):
        '''self.img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        self.mask_path = mask_path
        self.log = log
        self.procedure = procedure'''
        self.imgobj = imgobj
        self.img = cv2.rotate(imgobj.annotated_img, cv2.ROTATE_90_CLOCKWISE)
        #self.mask_path = mask_path
        self.procedure = procedure

        self.zoomed_img = []
        self.zoomxy = 0,0
        self.zoom_factor = 4
        self.height, self.width = self.img.shape [:2]
        

    def zoom_on_click(self, event, x, y, flags, param):
        #nonlocal zoomed_img
        if event == cv2.EVENT_LBUTTONDOWN:
            # Calculate coordinates for zoomed region
            zoom_size = 100  # Adjust zoom size as needed
            self.zoomxy = max(0, x - zoom_size), max(0, y - zoom_size)
            x1, y1 = self.zoomxy
            x2, y2 = min(self.img.shape[1], x + zoom_size), min(self.img.shape[0], y + zoom_size)

            # Extract zoomed region and resize
            zoomed_img0 = self.img[y1:y2, x1:x2]
            self.zoomed_img = cv2.resize(zoomed_img0, (0, 0), fx=self.zoom_factor, fy=self.zoom_factor)

            

    def reverse_augmentations(self, promptobj):
        polygon_vertices = promptobj.get_points()
        # Resize the zoomed image back to its original size
        #cropped_img = cv2.resize(self.zoomed_img, (0,0),fx=1/self.zoom_factor, fy=1/self.zoom_factor)
        #resized_h , resized_w = cropped_img.shape[:2]

        # Calculate the starting position in the original image
        xstart, ystart = self.zoomxy

        

        # Scale the polygon vertices back to the original image size
        scaled_pts = [(int(x/self.zoom_factor) + xstart, int(y/self.zoom_factor) + ystart) for x, y in polygon_vertices]
        vertsnp = np.array(scaled_pts).reshape((1, -1, 2))


        return  scaled_pts

    def polygon_to_mask(self, img_shape, polygon_vertices):
        # Create an empty mask
        mask = np.zeros(img_shape[:2], dtype=np.uint8)

        # Convert polygon vertices to NumPy array format
        pts = np.array([polygon_vertices], dtype=np.int32)

        # Fill the polygon with white color (255) in the mask
        cv2.fillPoly(mask, pts, 255)

        return mask
    
    def annot_img(self, mask, color):
        self.img[np.where(mask==255)] = color
        return self.img

        
    
    def __call__(self):

       
        mask_string = "maskpoly_"
        index = 0
        suffix = ".png"
        if self.procedure == "sam_prompting":
            predictor = self.imgobj.sam
        cv2.namedWindow("Annotations")
        cv2.setMouseCallback("Annotations", self.zoom_on_click, param=self)

        while True:
            self.zoomed_img = []
            print("Select a region to annotate by clicking on the image.")
            print("Press (f) if all masks in this image have been anotated.")
            print("Press (b) if you want to undo the previous mask.")
            while len(self.zoomed_img)==0:
                cv2.imshow("Annotations", self.img)
                key = cv2.waitKey(50) & 0xFF
                if key == ord('f'):
                    cv2.destroyAllWindows()
                    print("All masks have been annotated.")
                    self.imgobj.annotated_img = cv2.rotate(self.img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    self.imgobj.mask_collection, _= update_collections([m.mask for m in self.imgobj.masks], 
                                                                       self.imgobj.img) 
                    return self.imgobj
                elif key == ord('b'):
                    if len(self.imgobj.masks) == 0:
                        print("No masks to undo.")
                        continue
                    else:
                        self.imgobj.masks.pop()
                        _, new_annots = update_collections([m.mask for m in self.imgobj.masks], 
                                                         self.imgobj.img)
                        self.img = cv2.rotate(new_annots, cv2.ROTATE_90_CLOCKWISE)
                elif key == ord('x'):
                    cv2.destroyAllWindows()
                    raise KeyboardInterrupt()
            
            prompt = PromptRegistration(self.zoomed_img.copy())
            if self.procedure == "polygon_drawing":
                pts = prompt.draw_polygon()
            elif self.procedure == "sam_prompting":
                pts = prompt.draw_bbox()
            

            if len(pts) == 0:
                print("No mask created.")
                continue
          
            scaled_points = self.reverse_augmentations(prompt)
            if self.procedure == "polygon_drawing":
                mask = self.polygon_to_mask(self.img.shape[:2], scaled_points)
            elif self.procedure == "sam_prompting":
                mask = predictor.predict(bboxes=np.array(scaled_points).flatten()) 

            if len(mask) == 0:
                print("No mask created.")
                continue

            self.img = self.annot_img(mask, prompt.get_color())

            #self.log[f"{self.procedure}_selection"] = [index + 1]
            
            #saved_path = os.path.join(self.mask_path, mask_string + str(index) + suffix)
            upmask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            print('Created mask', upmask.shape)
            #cv2.imwrite(saved_path, upmask)
            #print("Saved to: ", saved_path)
            self.imgobj.masks.append(MaskData(upmask,self.procedure))
            index += 1
        
