import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(
    file_path,
    target_column,
    test_size=0.2,
    random_state=42,
    drop_columns=None,
    shuffle=True,
    sample_size=None,
):
    """
    Load and split the dataset into train and test sets.
    """
    drop_columns = [] if drop_columns is None else drop_columns
    df = pd.read_csv(file_path)
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)
    X = df.drop(drop_columns + [target_column], axis=1, errors='ignore')
    y = df[target_column]

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
    )
