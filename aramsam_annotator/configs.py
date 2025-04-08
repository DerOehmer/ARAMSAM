from dataclasses import dataclass, field


@dataclass
class SamConfigs:
    do_sam: bool = True
    gen: int = 1
    model_ckpt_p: str = "sam_vit_b_01ec64.pth"
    model_type: str = "vit_b"


@dataclass
class ImgTiles:
    do_tiling: bool = True
    tile_size: int = 640
    tile_overlap: float = 0.2


@dataclass
class SaveData:
    do_save: bool = True
    auto_save: bool = True
    save_dir: str = None
    save_masks: bool = False
    save_bboxes: bool = True
    style: str = "yolo"

    def __post_init__(self):
        if self.save_masks and self.save_bboxes:
            self.save_bboxes = False  # Fixed variable name
            print("Not implemented yet. Only saving masks")
        if self.auto_save and not self.do_save:
            self.do_save = True
            print("Auto save is enabled. Saving data will be enabled as well.")


@dataclass
class AramsamConfigs:
    sam_configs: SamConfigs = field(default_factory=SamConfigs)
    sam_backround_embedding: bool = False
    sam_amg: bool = False
    img_tiles: ImgTiles = field(default_factory=ImgTiles)
    yolo_model_ckpt_p: str = "KernelYOLO8x.pt"
    save_data: SaveData = field(default_factory=SaveData)
    class_json_p: str = "class.json"

    def __post_init__(self):
        if self.sam_amg and self.yolo_model_ckpt_p is not None:
            self.yolo_model_ckpt_p = None
            print(
                "AMG and YOLO model cannot be activated at the same time. Disabling YOLO model."
            )
