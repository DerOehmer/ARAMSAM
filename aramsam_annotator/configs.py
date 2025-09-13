from dataclasses import dataclass, field




@dataclass
class SamConfigs:
    do_sam: bool = True
    gen: int = 1 #2 #
    model_ckpt_p: str = "sam_vit_b_01ec64.pth"#"sam_vit_l_0b3195.pth"  #'sam2.1_hiera_small.pt'#
    model_type: str = "vit_b" #"vit_l"#'configs/sam2.1/sam2.1_hiera_s.yaml' #


@dataclass
class ImgTiles:
    do_tiling: bool = False
    tile_size: int = 640
    tile_overlap: float = 0.2


@dataclass
class SaveData:
    do_save: bool = True
    auto_save: bool = True
    save_dir: str | None = "output"
    save_masks: bool = True
    save_bboxes: bool = False
    bbox_style: str = "yolo"

    def __post_init__(self):
        if self.save_masks and self.save_bboxes:
            self.save_bboxes = False  # Fixed variable name
            print("Not implemented yet. Only saving masks")
        if self.auto_save and not self.do_save:
            self.do_save = True
            print("Auto save is enabled. Saving data will be enabled as well.")
        if self.bbox_style != "yolo":
            self.bbox_style = "yolo"
            print("Only YOLO bbox style is currently supported. Setting to 'yolo'.")


@dataclass
class AramsamConfigs:
    sam_configs: SamConfigs = field(default_factory=SamConfigs)
    sam_background_embedding: bool = True
    do_amg: bool = True

    img_tiles: ImgTiles = field(default_factory=ImgTiles)

    yolo_model_ckpt_p: str | None = None #"plant_count_yolo11x.pt"  #None#"yolov8x.pt"

    save_data: SaveData = field(default_factory=SaveData)
    class_dict: dict = field(default_factory=lambda: {
        "0": "ValidPlant",
        "1": "InvalidPlant",
        "2": "Shark"
    })

    def __post_init__(self):
        if self.do_amg and self.yolo_model_ckpt_p is not None:
            self.yolo_model_ckpt_p = None
            print(
                "AMG and YOLO model cannot be activated at the same time. Disabling YOLO model."
            )
        if self.sam_background_embedding and self.sam_configs.gen != 2:
            self.sam_background_embedding = False
            print(
                "Background embedding is currently not supported for SAM generation 1. Disabling background embedding."
            )
        
