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
from sklearn.model_selection import TimeSeriesSplit

BIKE_SHARING_METAADATA = {}

path_bike_sharing = f"{DATASET_PATH}/{DATASET_NAMES.bike_sharing.name}.csv"

# prepare dataset
X_train, X_test, y_train, y_test = load_data(
    file_path=path_bike_sharing,
    target_column="cnt",
    drop_columns=["instant", "dteday", "casual", "registered"],
    shuffle=False,
    random_state=RANDOM_STATE,
)
BIKE_SHARING_METAADATA[DATASET_VAR] = (X_train, X_test, y_train, y_test)
BIKE_SHARING_METAADATA[PROBLEM_TYPE_VAR] = PROBLEM_TYPES.regression.name

# problem description
BIKE_SHARING_METAADATA[PROBLEM_DESCRIPTION_VAR] = PROBLEM_DESCRIPTIONS[
    DATASET_NAMES.bike_sharing.name
]

# optimization parameters
BIKE_SHARING_METAADATA[METRIC_VAR] = METRICS.neg_mean_absolute_error.name
BIKE_SHARING_METAADATA[METRIC_FUNC_VAR] = mae
BIKE_SHARING_METAADATA[DIRECTION_VAR] = OPTIMIZATION_DIRECTIONS.minimize.name
BIKE_SHARING_METAADATA[OPTIMIZATION_CHECK_FUNC_VAR] = min
BIKE_SHARING_METAADATA[CV_VAR] = TimeSeriesSplit(n_splits=N_SPLITS, test_size=100)
