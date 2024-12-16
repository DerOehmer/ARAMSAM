import os
import numpy as np


def _string_in_list(string, lst):
    if string in lst:
        return lst.index(string)
    else:
        return -1


def find_used_ear_ids(base_dir):
    img_names = []
    img_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".jpg") and "_low_" in file:
                id_name = file.split(".jpg")[0]
                id_name = id_name.split("_annots")[0]
                str_index = _string_in_list(id_name, img_names)
                if str_index != -1:
                    print(
                        "Duplicate in: ", img_paths[str_index], os.path.join(root, file)
                    )
                img_names.append(id_name)
                img_paths.append(os.path.join(root, file))
        for dir in dirs:
            if "segment-anything-21" in dir or "venv" in dir:
                continue
            elif "_low_" in dir:
                id_name = dir.split("_annots")[0]
                str_index = _string_in_list(id_name, img_names)
                if str_index != -1:
                    print(
                        "Duplicate in: ", img_paths[str_index], os.path.join(root, file)
                    )
                img_names.append(id_name)
                img_paths.append(os.path.join(root, dir))
    return img_names


if __name__ == "__main__":
    fnames = find_used_ear_ids(os.getcwd())
    print(len(fnames))
    print(len(np.unique(fnames)))
    print(np.unique(fnames))
    np.save("used_ear_ids.npy", np.unique(fnames))
