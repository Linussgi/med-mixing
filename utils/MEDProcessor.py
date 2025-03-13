import numpy as np
import pandas as pd
import medeq
from .general_utils import create_function, split_df


class MEDProcessor:
    def __init__(self, sweep: str, tag: str, dim: str):
        """
        Initializes the MEDProcessor with sweep, tag, and dimension.

        Args:
            sweep (str): The sweep string to determine parameter names.
            tag (str): A tag used for file naming.
            dim (str): The dimension used in the dataset.
        """
        self.sweep = sweep
        self.tag = tag
        self.dim = dim

        self.p1_name, self.p2_name = self.sweep.split("-")

    def prepare_data(self, split_frac: float, seed=42):
        """
        Prepares input and output data for MED.

        Returns:
            tuple: (train_df, test_df, output, parameters)
        """
        in_path = f"k_csvs/{self.sweep}/fitted_k_values_{self.tag}.csv"
        df = pd.read_csv(in_path)

        train_df, test_df = split_df(df, split_frac, self.p1_name, self.p2_name, seed)
        traindf_truth = train_df[f"{self.dim} lacey k"]

        min_p1, max_p1 = df[self.p1_name].min(), df[self.p1_name].max()
        min_p2, max_p2 = df[self.p2_name].min(), df[self.p2_name].max()

        # Create parameter definitions for MED
        parameters = {
            "names": [self.p1_name, self.p2_name],
            "minimums": [min_p1, min_p2],
            "maximums": [max_p1, max_p2]
        }

        return train_df, test_df, traindf_truth, parameters

    def run_med_discovery(self, train_df: pd.DataFrame, output: pd.Series, parameters: dict):
        """
        Runs MED symbolic regression to discover equations.

        Args:
            train_df (pd.DataFrame): Training dataset.
            output (pd.Series): Output variable for MED.
            parameters (dict): Dictionary containing parameter names and bounds.
        """
        # Create MED parameters
        med_params = medeq.create_parameters(
            parameters["names"],
            minimums=parameters["minimums"],
            maximums=parameters["maximums"]
        )

        # Create MED object
        med = medeq.MED(med_params, response_names=output.name, seed=123)

        # Add data
        med.augment(train_df[parameters["names"]], output)

        # Save results
        med.save(f"med_{self.sweep}_{self.tag}_{self.dim}")

        # Discover equations
        med.discover(
            binary_operators=["+", "-", "*", "/", "^"],
            constraints={"^": (-1, 1)},
            unary_operators=["exp", "log"],
            equation_file=f"med_{self.sweep}_{self.tag}_{self.dim}/hall_of_fame.csv"
        )

    def process_results(self, test_df, tmp_path=None):
        """
        Processes MED results by evaluating equations on test data and computing relative errors.

        Args:
            test_df (pd.DataFrame): Testing dataset.

        Returns:
            pd.DataFrame: A DataFrame containing actual values and relative errors for each complexity.
        """
        eq_path = tmp_path if tmp_path else f"med_{self.sweep}_{self.tag}_{self.dim}/hall_of_fame.csv"
        df_equations = pd.read_csv(eq_path)

        df_equations["Function"] = df_equations.apply(
            lambda row: create_function(row["Equation"], self.p1_name, self.p2_name), axis=1
        )

        test_df = test_df.sort_values(by=[self.p1_name, self.p2_name])

        results = []
        for _, test_row in test_df.iterrows():
            result_row = {
                f"{self.p1_name}_value": test_row[self.p1_name],
                f"{self.p2_name}_value": test_row[self.p2_name],
                f"actual_k_{self.dim}": test_row[f"{self.dim} lacey k"]
            }

            for _, eq_row in df_equations.iterrows():
                func = eq_row["Function"]
                complexity = eq_row["Complexity"]

                if func:
                    pred = func(test_row[self.p1_name], test_row[self.p2_name])
                    actual = test_row[f"{self.dim} lacey k"]
                    
                    err = np.abs(actual - pred) / (np.abs(actual) + 1e-8)
                    result_row[f"Complexity {complexity} k{self.dim} p err"] = err

            results.append(result_row)

        return pd.DataFrame(results)

    def get_complexity_equation(self, complexity, tmp_path=None):
        """
        Fetches the equation from the hall_of_fame.csv file based on the given complexity.

        Args:
            complexity (int): The complexity level to find the equation for.

        Returns:
            str: The equation corresponding to the given complexity, or None if not found.
        """
        eq_path = tmp_path if tmp_path else f"med_{self.sweep}_{self.tag}_{self.dim}/hall_of_fame.csv"
        df_equations = pd.read_csv(eq_path)

        equation_row = df_equations[df_equations["Complexity"] == complexity]

        return equation_row["Equation"].values[0] if not equation_row.empty else None
