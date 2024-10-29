import pandas as pd


def concat_csvs(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    img_paths = set()
    for df in df_list:
        unique_imgs = set(df["img_dir"].unique())
        if img_paths & unique_imgs:
            raise ValueError(
                f"Duplicate image paths in dataframes{img_paths & unique_imgs}"
            )
        else:
            img_paths.update(unique_imgs)
    return pd.concat(df_list, ignore_index=True)


def main():
    df1 = pd.read_csv("Exp32/Sam1_VitB_5-10.csv")
    df2 = pd.read_csv("test0-2.csv")

    df_concat = concat_csvs([df1, df2])

    print(len(df1), len(df2), len(df_concat))
    print(len(df1) + len(df2))
    df_concat.to_csv("Exp32/Sam1_VitB_0-2_5-10.csv", index=False)


if __name__ == "__main__":
    main()
