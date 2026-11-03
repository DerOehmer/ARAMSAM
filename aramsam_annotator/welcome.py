"""Startup settings form; no model loading or file writes happen in this widget."""
from dataclasses import asdict, fields, is_dataclass

import yaml
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)

from aramsam_annotator.configs import AramsamConfigs, ImgTiles, SamConfigs, SaveData


LABELS = {
    "do_sam": "Enable SAM", "gen": "SAM generation", "model_ckpt_p": "Model checkpoint",
    "model_type": "Model type / SAM2 config", "bpe_path": "SAM3 tokenizer vocabulary",
    "confidence_threshold": "Text-prompt confidence", "use_gpu": "Use GPU",
    "sam_background_embedding": "Background image embedding", "do_amg": "Automatic mask proposals",
    "mask_propagation": "Mask propagation", "yolo_model_ckpt_p": "YOLO checkpoint (optional)",
    "do_tiling": "Split images into tiles", "tile_size": "Tile size (pixels)",
    "tile_overlap": "Tile overlap", "do_save": "Save annotations", "auto_save": "Auto Save",
    "save_dir": "Output directory", "save_masks": "Save masks", "mask_style": "Mask format",
    "save_bboxes": "Save bounding boxes", "bbox_style": "Bounding-box format",
}


class SettingsDoubleSpinBox(QDoubleSpinBox):
    def textFromValue(self, value):
        text = self.locale().toString(value, 'f', self.decimals())
        return text.rstrip('0').rstrip(self.locale().decimalPoint()) if self.decimals() else text


class WelcomeWindow(QDialog):
    load_requested = pyqtSignal(str)

    def __init__(self, configs, config_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to ARAMSAM")
        self.setModal(True)
        self.initial_values = asdict(configs)
        self.controls = {}
        self.nullable = set()
        self.original_values = {}
        self.edited_numbers = set()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        title = QLabel("Welcome to ARAMSAM")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)
        intro = QLabel("Review settings if needed, then load an image or a folder to begin.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._add_section("General", configs, "")
        self._add_section("SAM", configs.sam_configs, "sam_configs")
        self._add_section("Image Tiling", configs.img_tiles, "img_tiles")
        self._add_section("Saving", configs.save_data, "save_data")
        classes = QWidget()
        classes_layout = QVBoxLayout(classes)
        explanation = QLabel('Class IDs and names, one per line. Example: 0: Seed\nLeave {} for no classes.')
        explanation.setWordWrap(True)
        classes_layout.addWidget(explanation)
        self.classes_editor = QPlainTextEdit(yaml.safe_dump(configs.class_dict, sort_keys=False, allow_unicode=True))
        self.classes_editor.setAccessibleName("Class IDs and names")
        classes_layout.addWidget(self.classes_editor)
        self.controls["class_dict"] = self.classes_editor
        self.tabs.addTab(classes, "Classes")
        destination = QLabel(f"Changed settings will be saved to: {config_path}")
        destination.setWordWrap(True)
        destination.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(destination)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setTextFormat(Qt.TextFormat.PlainText)
        self.error_label.setStyleSheet("color: #ff8e8e;")
        self.error_label.hide()
        layout.addWidget(self.error_label)
        buttons = QHBoxLayout()
        self.quit_button = QPushButton("Quit")
        self.image_button = QPushButton("Load Image")
        self.folder_button = QPushButton("Load Folder")
        for button in (self.quit_button, self.image_button, self.folder_button):
            button.setAutoDefault(False)
            button.setMinimumHeight(40)
            button.setMinimumWidth(120)
        self.quit_button.setStyleSheet("QPushButton { background-color: #b83b45; color: white; } QPushButton:hover { background-color: #cf4b55; }")
        for button in (self.image_button, self.folder_button):
            button.setStyleSheet("QPushButton { background-color: #2468c9; color: white; } QPushButton:hover { background-color: #347ce0; }")
        buttons.addWidget(self.quit_button)
        buttons.addStretch()
        buttons.addWidget(self.image_button)
        buttons.addWidget(self.folder_button)
        layout.addLayout(buttons)
        self.quit_button.clicked.connect(self.reject)
        self.image_button.clicked.connect(lambda: self.load_requested.emit("image"))
        self.folder_button.clicked.connect(lambda: self.load_requested.emit("folder"))
        self.controls["sam_configs.gen"].currentIndexChanged.connect(self._generation_changed)
        self.controls["save_data.save_masks"].toggled.connect(self._masks_changed)
        self.controls["save_data.save_bboxes"].toggled.connect(self._boxes_changed)
        self.controls["save_data.auto_save"].toggled.connect(self._auto_save_changed)
        self.controls["save_data.do_save"].toggled.connect(self._saving_changed)
        available = self.screen().availableGeometry()
        self.resize(min(760, available.width() - 40), min(680, available.height() - 60))

    def _add_section(self, title, settings, prefix):
        content = QWidget()
        form = QFormLayout(content)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(12)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for field in fields(settings):
            value = getattr(settings, field.name)
            if is_dataclass(value) or field.name == "class_dict":
                continue
            key = f"{prefix}.{field.name}" if prefix else field.name
            if field.type is bool:
                control = QCheckBox()
                control.setChecked(value)
            elif key == "sam_configs.gen":
                control = QComboBox()
                for generation in (1, 2, 3):
                    control.addItem(f"SAM {generation}", generation)
                control.setCurrentIndex(control.findData(value))
            elif field.type is int:
                control = QSpinBox()
                control.setRange(1, 1024 if field.name == "tile_size" else 1000000)
                control.setValue(value)
            elif field.type is float:
                control = SettingsDoubleSpinBox()
                control.setRange(0, 0.99 if field.name == "tile_overlap" else 1)
                control.setDecimals(10)
                control.setSingleStep(0.05)
                control.setValue(value)
                control.valueChanged.connect(lambda _, key=key: self.edited_numbers.add(key))
            else:
                control = QLineEdit("" if value is None else str(value))
                if value is None or "None" in str(field.type):
                    self.nullable.add(key)
                    control.setPlaceholderText("Automatic / not set")
            control.setToolTip(key)
            control.setAccessibleName(LABELS.get(field.name, field.name))
            self.controls[key] = control
            self.original_values[key] = value
            form.addRow(LABELS.get(field.name, field.name), control)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self.tabs.addTab(scroll, title)

    def _generation_changed(self):
        defaults = {
            1: ("sam_vit_b_01ec64.pth", "vit_b"),
            2: ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
            3: ("sam3.pt", ""),
        }
        generation = self.controls["sam_configs.gen"].currentData()
        for index, name in enumerate(("model_ckpt_p", "model_type")):
            control = self.controls[f"sam_configs.{name}"]
            if not control.text() or control.text() in {value[index] for value in defaults.values()}:
                control.setText(defaults[generation][index])

    def _masks_changed(self, checked):
        if checked:
            self.controls["save_data.save_bboxes"].setChecked(False)

    def _boxes_changed(self, checked):
        if checked:
            self.controls["save_data.save_masks"].setChecked(False)
            self.controls["save_data.bbox_style"].setText("yolo")

    def _auto_save_changed(self, checked):
        if checked:
            self.controls["save_data.do_save"].setChecked(True)

    def _saving_changed(self, checked):
        if not checked:
            self.controls["save_data.auto_save"].setChecked(False)

    def read_configs(self):
        values = {}
        for key, control in self.controls.items():
            if key == "class_dict":
                continue
            if isinstance(control, QCheckBox):
                value = control.isChecked()
            elif isinstance(control, QComboBox):
                value = control.currentData()
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                value = (self.original_values[key] if isinstance(control, QDoubleSpinBox)
                         and key not in self.edited_numbers else control.value())
            else:
                value = control.text().strip()
                if not value and (key in self.nullable or key in ("sam_configs.model_type", "sam_configs.model_ckpt_p")):
                    value = None
            if "." in key:
                group, name = key.split(".")
                values.setdefault(group, {})[name] = value
            else:
                values[key] = value
        try:
            classes = yaml.safe_load(self.classes_editor.toPlainText())
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid class mapping: {error}") from error
        if not isinstance(classes, dict):
            raise ValueError("Classes must be a mapping of numeric IDs to names, or {}.")
        ids = set()
        for key, name in classes.items():
            try:
                class_id = int(str(key))
            except ValueError as error:
                raise ValueError("Class IDs must be nonnegative integers.") from error
            if class_id < 0 or class_id in ids or not isinstance(name, str) or not name.strip():
                raise ValueError("Classes need unique nonnegative integer IDs and nonempty names.")
            ids.add(class_id)
        if values["do_amg"] and values["yolo_model_ckpt_p"]:
            raise ValueError("Choose automatic mask proposals or a YOLO checkpoint, not both.")
        if values["do_amg"] and not values["sam_configs"]["do_sam"]:
            raise ValueError("Enable SAM to use automatic mask proposals.")
        if values["sam_configs"]["gen"] not in (1, 2, 3):
            raise ValueError("Choose SAM generation 1, 2, or 3.")
        if not 0 <= values["img_tiles"]["tile_overlap"] < 1:
            raise ValueError("Tile overlap must be between 0 (inclusive) and 1 (exclusive).")
        if values["sam_configs"]["do_sam"] and values["sam_configs"]["gen"] == 3 and not values["use_gpu"]:
            raise ValueError("SAM3 requires Use GPU. Enable it or select SAM1/SAM2.")
        return AramsamConfigs(
            **{key: value for key, value in values.items() if key not in ("sam_configs", "img_tiles", "save_data")},
            sam_configs=SamConfigs(**values["sam_configs"]), img_tiles=ImgTiles(**values["img_tiles"]),
            save_data=SaveData(**values["save_data"]), class_dict=classes,
        )

    def show_error(self, message):
        self.error_label.setText(message)
        self.error_label.show()
