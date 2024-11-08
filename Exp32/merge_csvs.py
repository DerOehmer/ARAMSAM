import pandas as pd
import os


def concat_csvs(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    img_paths = set()
    df_len_control = 0
    df_col_n = None
    for df in df_list:
        if df_col_n is not None:
            assert (
                len(df.columns) == df_col_n
            ), "Inequal number of columns in dataframes"
        df_col_n = len(df.columns)
        unique_imgs = set(df["img_dir"].unique())
        if img_paths & unique_imgs:
            raise ValueError(
                f"Duplicate image paths in dataframes{img_paths & unique_imgs}"
            )
        else:
            img_paths.update(unique_imgs)
            df_len_control += len(df)

    df_concat = pd.concat(df_list, ignore_index=True)
    assert len(df_concat.columns) == df_col_n, "Inequal number of columns in dataframes"
    assert (
        len(df_concat) == df_len_control
    ), "Inequal length of concatenated df and predicted length"
    print(
        f"Concatenated df length ({len(df_concat)}) is equal to estimated length({df_len_control})"
    )
    print(f"Concatenated df contains result data of {len(img_paths)} images")

    return df_concat


def _check_paths(path_list: list[str]) -> None:
    for path in path_list:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist")


def get_df_list(paths: list[str]) -> list[pd.DataFrame]:
    _check_paths(paths)
    return [pd.read_csv(path) for path in paths]


def save_df(df: pd.DataFrame, path: str) -> None:
    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists")
    df.to_csv(path, index=False)


def main():
    input_paths = [
        "Exp32/Sam1_VitH_0-5.csv",
        "Exp32/Sam1_VitH_5-8.csv",
        "Exp32/Sam1_VitH_8-10.csv",
    ]
    dest_path = "Exp32/Sam1_VitH.csv"

    dfs = get_df_list(input_paths)
    df_concat = concat_csvs(dfs)

    save_df(df_concat, dest_path)


if __name__ == "__main__":
    main()
