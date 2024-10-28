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

ENERGY_METADATA = {}

path_energy = f"{DATASET_PATH}/{DATASET_NAMES.energy.name}.csv"

# Load your dataset
X_train, X_test, y_train, y_test = load_data(
    file_path=path_energy,
    target_column="Appliances",
    drop_columns=["date"],
    shuffle=False,
    random_state=RANDOM_STATE,
)
ENERGY_METADATA[DATASET_VAR] = (X_train, X_test, y_train, y_test)
ENERGY_METADATA[PROBLEM_TYPE_VAR] = PROBLEM_TYPES.regression.name

# problem description
ENERGY_METADATA[PROBLEM_DESCRIPTION_VAR] = PROBLEM_DESCRIPTIONS[
    DATASET_NAMES.energy.name
]

# optimization parameters
ENERGY_METADATA[METRIC_VAR] = METRICS.neg_mean_absolute_error.name
ENERGY_METADATA[METRIC_FUNC_VAR] = mae
ENERGY_METADATA[DIRECTION_VAR] = OPTIMIZATION_DIRECTIONS.minimize.name
ENERGY_METADATA[OPTIMIZATION_CHECK_FUNC_VAR] = min
ENERGY_METADATA[CV_VAR] = TimeSeriesSplit(n_splits=N_SPLITS, test_size=100)
