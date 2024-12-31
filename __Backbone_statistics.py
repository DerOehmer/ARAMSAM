import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations


def create_pvalue_matrix_from_dataframe(
    df: pd.DataFrame,
    required_columns: list = ["Weights path", "Mask N", "Dice per mask"],
    sign_level: float = 0.05,
) -> pd.DataFrame:
    """
    Computes a p-value matrix comparing all pairs of models using paired t-tests.

    Parameters:
    - df: pandas DataFrame with columns #[required_columns[0], required_columns[1], required_columns[2]]

    Returns:
    - p_value_matrix: pandas DataFrame containing p-values for each model pair
    """

    # Get the list of unique models
    models = df[required_columns[0]].unique()

    # Initialize the p-value matrix as a DataFrame
    p_value_matrix = pd.DataFrame(index=models, columns=models, dtype=float)

    # Iterate over all combinations of models
    for model1, model2 in combinations(models, 2):
        # Extract data for the two models
        data1 = df[df[required_columns[0]] == model1][
            [required_columns[1], required_columns[2]]
        ]
        data2 = df[df[required_columns[0]] == model2][
            [required_columns[1], required_columns[2]]
        ]

        merged_data = pd.merge(
            data1, data2, on=required_columns[1], suffixes=("_1", "_2")
        )

        # Check if there are enough paired observations
        if len(merged_data) < 2:
            p_value = np.nan  # Not enough data to perform t-test
        else:
            # Perform the paired t-test
            t_statistic, p_value = stats.ttest_rel(
                merged_data[f"{required_columns[2]}_1"],
                merged_data[f"{required_columns[2]}_2"],
                nan_policy="omit",
            )

        # Store the p-value in the matrix
        p_value_matrix.loc[model1, model2] = p_value
        p_value_matrix.loc[model2, model1] = p_value  # Symmetric matrix

    # Fill the diagonal with NaNs or zeros since a model compared to itself is not informative
    np.fill_diagonal(p_value_matrix.values, np.nan)

    return p_value_matrix


if __name__ == "__main__":
    dest_dir = "BackboneExperimentData/MaizeEarSignMatrices/"
    metrics_df = pd.read_csv("BackboneExperimentData/MaizeEar_results_per_mask.csv")
    sam_gens = metrics_df["Sam generation"].unique()

    for sam_gen in sam_gens:
        sam_gen_df = metrics_df[metrics_df["Sam generation"] == sam_gen]
        for metric in ["Dice per mask", "Gdice per mask", "IoU per mask"]:
            p_value_matrix = create_pvalue_matrix_from_dataframe(
                sam_gen_df, required_columns=["Weights path", "Mask N", metric]
            )
            p_value_matrix.to_csv(
                f"{dest_dir}{sam_gen}_{metric}_significance_matrix.csv", index=True
            )
