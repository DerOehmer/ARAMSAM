import dataclasses

from aramsam_annotator.app import App
from aramsam_annotator.mask_visualizations import MaskVisualizationData
from aramsam_annotator.configs import AramsamConfigs


def create_vis_options() -> tuple[list[str]]:
    vis_options = [field.name for field in dataclasses.fields(MaskVisualizationData)]
    default_vis_options = [
        "img",
        "img_sam_preview",
        "bbox_img_cnt",
        "mask_collection_cnt",
    ]

    for default_vis_option in default_vis_options:
        if default_vis_option not in vis_options:
            if len(vis_options) >= 4:
                default_vis_options = vis_options[:4]
            else:
                default_vis_options = [vis_options[0] for _ in range(4)]

    current_options = default_vis_options

    return vis_options, default_vis_options, current_options


def main():
    vis_options, default_vis_options, current_options = create_vis_options()
    ui_options = {
        "layout_settings_options": {
            "options": vis_options,
            "default": default_vis_options,
            "current": current_options,
        }
    }
    confs = AramsamConfigs()
    app = App(ui_options=ui_options, configs=confs)
    print("app")
    app.run()


if __name__ == "__main__":
    main()
