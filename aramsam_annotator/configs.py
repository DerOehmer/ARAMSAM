from dataclasses import dataclass


@dataclass
class SamConfigs:
    gen: int = 1
    model_ckpt_p: str = "sam_vit_b_01ec64.pth"
    model_type: str = "vit_b"


@dataclass
class ImgTiles:
    do_tiling: bool = True
    tile_size: int = 640
    tile_overlap: float = 0.2


@dataclass
class AramsamConfigs:
    sam_configs: SamConfigs = SamConfigs()
    sam_backround_embedding: bool = True
    sam_amg: bool = False
    img_tiles: ImgTiles = ImgTiles()
    yolo_model_ckpt_p: str = "KernelYOLO8x.pt"

    def __post_init__(self):
        if self.sam_amg and self.yolo_model_ckpt_p is not None:
            self.yolo_model_ckpt_p = None
            print(
                "AMG and YOLO model cannot be activated at the same time. Disabling YOLO model."
            )
