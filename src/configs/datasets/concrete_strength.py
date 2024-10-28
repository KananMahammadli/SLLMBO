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
from sklearn.model_selection import KFold

CONCRETE_STRENGTH_METADATA = {}

path_cement__strength = f"{DATASET_PATH}/{DATASET_NAMES.concrete_strength.name}.csv"

# Load your dataset
X_train, X_test, y_train, y_test = load_data(
    file_path=path_cement__strength,
    target_column="strength",
    drop_columns=["age", "fine_aggregate"],
    shuffle=True,
    random_state=RANDOM_STATE,
)
CONCRETE_STRENGTH_METADATA[DATASET_VAR] = (X_train, X_test, y_train, y_test)
CONCRETE_STRENGTH_METADATA[PROBLEM_TYPE_VAR] = PROBLEM_TYPES.regression.name

# problem description
CONCRETE_STRENGTH_METADATA[PROBLEM_DESCRIPTION_VAR] = PROBLEM_DESCRIPTIONS[
    DATASET_NAMES.concrete_strength.name
]

# optimization parameters
CONCRETE_STRENGTH_METADATA[METRIC_VAR] = METRICS.neg_mean_absolute_error.name
CONCRETE_STRENGTH_METADATA[METRIC_FUNC_VAR] = mae
CONCRETE_STRENGTH_METADATA[DIRECTION_VAR] = OPTIMIZATION_DIRECTIONS.minimize.name
CONCRETE_STRENGTH_METADATA[OPTIMIZATION_CHECK_FUNC_VAR] = min
CONCRETE_STRENGTH_METADATA[CV_VAR] = KFold(
    n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
)
