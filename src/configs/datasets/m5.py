from configs.config_enum import (
    DATASET_NAMES,
    METRICS,
    OPTIMIZATION_DIRECTIONS,
    PROBLEM_TYPES,
)
from configs.config_problem_descriptions import (
    PROBLEM_DESCRIPTIONS,
)
from data_loader import load_data
from configs.config_params import (
    RANDOM_STATE,
    N_SPLITS,
    CV_VAR,
    DATASET_PATH,
    DATASET_VAR,
    DIRECTION_VAR,
    METRIC_FUNC_VAR,
    METRIC_VAR,
    OPTIMIZATION_CHECK_FUNC_VAR,
    PROBLEM_DESCRIPTION_VAR,
    PROBLEM_TYPE_VAR,
)

from sklearn.metrics import mean_absolute_error as mae
from mlxtend.evaluate.time_series import GroupTimeSeriesSplit
import pandas as pd

M5_METADATA = {}

path_m5 = f"{DATASET_PATH}/{DATASET_NAMES.m5.name}.csv"
ph = 28

# load dataset
X_train, X_test, y_train, y_test = load_data(
    file_path=path_m5,
    target_column="sales",
    test_size=5040,
    random_state=RANDOM_STATE,
    drop_columns=[],
    shuffle=False,
    sample_size=None,
)
cv_Splitter = GroupTimeSeriesSplit(
    n_splits=N_SPLITS,
    test_size=ph,
    shift_size=ph,
    window_type="rolling",
)
groups = pd.factorize(X_train["date"])[0]
cv = list(cv_Splitter.split(X_train, y_train, groups=groups))

# drop unused columns
unused_cols = [
    "id",
    "date",
    "d",
]
cat_cols = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]

for df in [X_train, X_test]:
    df.drop(columns=unused_cols, inplace=True)
    df[cat_cols] = df[cat_cols].astype("category")


M5_METADATA[DATASET_VAR] = (X_train, X_test, y_train, y_test)
M5_METADATA[PROBLEM_TYPE_VAR] = PROBLEM_TYPES.regression.name

# problem description
M5_METADATA[PROBLEM_DESCRIPTION_VAR] = PROBLEM_DESCRIPTIONS[DATASET_NAMES.m5.name]

# optimization parameters
M5_METADATA[METRIC_VAR] = METRICS.neg_mean_absolute_error.name
M5_METADATA[METRIC_FUNC_VAR] = mae
M5_METADATA[DIRECTION_VAR] = OPTIMIZATION_DIRECTIONS.minimize.name
M5_METADATA[OPTIMIZATION_CHECK_FUNC_VAR] = min
M5_METADATA[CV_VAR] = cv
