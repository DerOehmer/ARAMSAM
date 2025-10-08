from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os
import yaml




@dataclass
class SamConfigs:
    do_sam: bool = True
    gen: int = 1 #2 #
    model_ckpt_p: str = "sam_vit_b_01ec64.pth"#"sam_vit_l_0b3195.pth"  #'sam2.1_hiera_small.pt'#
    model_type: str = "vit_b" #"vit_l"#'configs/sam2.1/sam2.1_hiera_s.yaml' #


@dataclass
class ImgTiles:
    do_tiling: bool = True
    tile_size: int = 640
    tile_overlap: float = 0.2


@dataclass
class SaveData:
    do_save: bool = True
    auto_save: bool = True
    save_dir: str | None = "output"
    save_masks: bool = True # 
    mask_style: str = "yolo" # default, yolo, 
    save_bboxes: bool = False
    bbox_style: str = "yolo" # default, yolo, ""

    def __post_init__(self):
        if self.save_masks and self.save_bboxes:
            self.save_bboxes = False  # Fixed variable name
            print("Not implemented yet. Only saving masks")

        if not self.save_bboxes:
            self.bbox_style = ""
        if self.auto_save and not self.do_save:
            self.do_save = True
            print("Auto save is enabled. Saving data will be enabled as well.")
        if self.save_bboxes and self.bbox_style != "yolo":
            self.bbox_style = "yolo"
            print("Only YOLO bbox style is currently supported. Setting to 'yolo'.")


@dataclass
class AramsamConfigs:
    sam_configs: SamConfigs = field(default_factory=SamConfigs)
    sam_background_embedding: bool = True
    do_amg: bool = False

    img_tiles: ImgTiles = field(default_factory=ImgTiles)

    yolo_model_ckpt_p: str | None = None #"plant_count_yolo11x.pt"  #None#"yolov8x.pt"

    save_data: SaveData = field(default_factory=SaveData)
    class_dict: dict = field(default_factory=lambda: {
        "0": "Larvae",
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


def load_configs_from_yaml(yaml_path: str | Path) -> "AramsamConfigs":
    """
    Load configuration from a YAML file and return an AramsamConfigs instance.

    The loader will only override keys provided in the YAML file and will keep
    the dataclass defaults for any missing fields. Nested configuration
    sections are supported for `sam_configs`, `img_tiles`, and `save_data`.

    Raises:
        ImportError: if PyYAML is not installed.
        FileNotFoundError: if the yaml_path does not exist.
        ValueError: if the YAML cannot be parsed into a dictionary.
    """

    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a YAML mapping (dictionary) at the top level.")

    # Build nested dataclasses using values from YAML where provided, else defaults
    sam_cfg = SamConfigs(**(data.get("sam_configs") or {}))
    img_tiles = ImgTiles(**(data.get("img_tiles") or {}))
    save_data = SaveData(**(data.get("save_data") or {}))

    # Top-level simple fields
    sam_background_embedding = data.get("sam_background_embedding", True)
    do_amg = data.get("do_amg", False)
    yolo_model_ckpt_p = data.get("yolo_model_ckpt_p", None)
    class_dict = data.get("class_dict", {"0": "Larvae"})

    # Create the AramsamConfigs instance which will run its post-init checks
    configs = AramsamConfigs(
        sam_configs=sam_cfg,
        sam_background_embedding=sam_background_embedding,
        do_amg=do_amg,
        img_tiles=img_tiles,
        yolo_model_ckpt_p=yolo_model_ckpt_p,
        save_data=save_data,
        class_dict=class_dict,
    )

    return configs


def find_and_load_configs(yaml_filename: str = "configs.yaml") -> "AramsamConfigs":
    """
    Helper that looks for a `configs.yaml` file in the current working directory
    and the repository root (two levels up from this module), and loads it if
    found. If not found, returns the default AramsamConfigs instance.
    """
    # check cwd first
    cwd_path = Path(os.getcwd()) / yaml_filename
    if cwd_path.exists():
        return load_configs_from_yaml(cwd_path)

    # check repo root (two levels up from this file)
    repo_root = Path(__file__).resolve().parents[1]
    repo_path = repo_root / yaml_filename
    if repo_path.exists():
        return load_configs_from_yaml(repo_path)

    # fallback to defaults
    return AramsamConfigs()
        
