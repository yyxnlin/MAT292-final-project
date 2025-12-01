import pandas as pd
import os 


def aggregate_and_filter_outputs(data_dir, loss_col='loss', threshold=0.1):
    parquet_files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
    dfs = []
    for f in parquet_files:
        df_file = pd.read_parquet(os.path.join(data_dir, f))
        
        # Keep only rows with loss < 0.1
        if loss_col in df_file.columns:
            df_file = df_file[df_file[loss_col] < threshold]
        
        # attach "recording" column used for filtering by patient later for testing set
        df_file['recording'] = f
        dfs.append(df_file)

    all_df = pd.concat(dfs, ignore_index=True)
    return all_df

def balanced_resample(df, col="symbol_binary", random_state=0):
    """
    Return a balanced dataframe by sampling an equal number of rows 
    from each category in df[col]. The sample size is equal to the 
    smallest class size.
    """
    # List all category names
    categories = df[col].unique()

    # Find the smallest class size
    min_size = df[col].value_counts().min()

    # Sample min_size rows from each category
    samples = []
    counts = {}

    for c in categories:
        df_c = df[df[col] == c]
        df_sample = df_c.sample(n=min_size, random_state=random_state)
        samples.append(df_sample)
        counts[c] = len(df_sample)

    # Combine & shuffle
    df_balanced = (
        pd.concat(samples)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    return df_balanced, counts
