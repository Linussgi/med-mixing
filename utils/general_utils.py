import pandas as pd
import sympy as sp


def get_param_values(text: str) -> list[tuple[str, float]]:
    """Extracts parameter names and values from a given text."""
    p1_name, p1_value, p2_name, p2_value = text.split("_")
    return (p1_name, float(p1_value)), (p2_name, float(p2_value))


def get_study_names(df: pd.DataFrame, frag=None) -> list[str]:
    """Extracts study names from """
    studies = set()

    for col in df.columns[2:]:
        name = col.split()[0]   
        
        if (frag and frag in name) or (not frag):
            studies.add(name)

    return list(studies)


def create_function(equation_str: str, p1_name: str, p2_name: str) -> optional[callable], None:
    equation_str = equation_str.replace("^", "**")

    x, y = sp.symbols(f"{p1_name} {p2_name}")

    expr = sp.sympify(equation_str, locals={"exp": sp.exp, "log": sp.log})

    func = sp.lambdify((x, y), expr, "numpy")

    print(f"Equation: {equation_str}")
    return func

def split_df(df: pd.DataFrame, split_frac: float, p1_name, p2_name, seed=42) -> tuple[pd.DataFrame]:
    df[p1_name] = df["study name"].apply(lambda x: get_param_values(x)[0][1])
    df[p2_name] = df["study name"].apply(lambda x: get_param_values(x)[1][1])

    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(split_frac * len(df_shuffled))
    train_df, test_df = df_shuffled.iloc[:split_idx], df_shuffled.iloc[split_idx:]

    return train_df, test_df