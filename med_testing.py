from utils.MEDProcessor import MEDProcessor
import pandas as pd


sweep = "amp-fill"
tag = "r1"
dim = "r"

splits = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
seed = 40

splits = [0.7]

dfs = []
best_equations = []

for split in splits:
    med_study = MEDProcessor(sweep, tag, dim)

    train_df, test_df, output, parameters = med_study.prepare_data(split, seed)

    med_study.run_med_discovery(train_df, output, parameters)

    if len(splits) == 1:
        hof = input("splits len is 1. Confirm hall_of_fame file has been moved (y/n): ")

        if hof != "y":
            raise ValueError("User Cancelled")
        
        tmp_path = None
    else:
        tmp_path = input("tmp path: ").strip()

    df_results = med_study.process_results(test_df, tmp_path)

    # Create the "average error" row for each complexity equation
    avg_row = df_results.filter(like="Complexity").mean().to_dict()

    avg_row[df_results.columns[0]] = "Average"  
    avg_row[df_results.columns[1]] = None
    avg_row[df_results.columns[2]] = None

    df_results = pd.concat([df_results, pd.DataFrame([avg_row])], ignore_index=True)
    dfs.append(df_results)

    best_column = min(avg_row, key=lambda x: avg_row[x] if x.startswith("Complexity") else float("inf"))
    
    best_complexity = int(best_column.split()[1])
    best_equation = med_study.get_complexity_equation(best_complexity, tmp_path)

    best_equations.append({
        "Split Value": split,
        "Complexity": best_complexity,
        "Equation": best_equation,
        "Average Loss": avg_row[best_column]
    })

if len(splits) == 1:
    df_results.to_csv(f"med_{sweep}_{tag}_{dim}/{tag}_{dim}_med.csv")

    print(f"Med results saved to 'med_{sweep}_{tag}_{dim}/{tag}_{dim}_med.csv'")
else:
    best_eq_df = pd.DataFrame(best_equations)
    best_eq_df.to_csv(f"med_{sweep}_{tag}_{dim}/best_equations_{sweep}_{tag}_{dim}.csv", index=False)

    print(f"Best equations saved to 'med_{sweep}_{tag}_{dim}/best_equations_{sweep}_{tag}_{dim}.csv'")