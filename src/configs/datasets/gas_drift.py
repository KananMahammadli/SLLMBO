from sklearn.preprocessing import LabelEncoder
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

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

GAS_DRIFT_METADATA = {}

path_gas_drift = f"{DATASET_PATH}/{DATASET_NAMES.gas_drift.name}.csv"

# prepare dataset
X_train, X_test, y_train, y_test = load_data(
    file_path=path_gas_drift,
    target_column="Class",
    drop_columns=["id"],
    random_state=RANDOM_STATE,
    shuffle=True,
)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
GAS_DRIFT_METADATA[DATASET_VAR] = (
    X_train.iloc[:, :5],
    X_test.iloc[:, :5],
    y_train,
    y_test,
)
GAS_DRIFT_METADATA[PROBLEM_TYPE_VAR] = PROBLEM_TYPES.classification.name

# problem description
GAS_DRIFT_METADATA[PROBLEM_DESCRIPTION_VAR] = PROBLEM_DESCRIPTIONS[
    DATASET_NAMES.gas_drift.name
]

# optimization parameters
GAS_DRIFT_METADATA[METRIC_VAR] = METRICS.f1_weighted.name
GAS_DRIFT_METADATA[METRIC_FUNC_VAR] = lambda y_true, y_pred: f1_score(
    y_true, y_pred, average="weighted"
)
GAS_DRIFT_METADATA[DIRECTION_VAR] = OPTIMIZATION_DIRECTIONS.maximize.name
GAS_DRIFT_METADATA[OPTIMIZATION_CHECK_FUNC_VAR] = max
GAS_DRIFT_METADATA[CV_VAR] = StratifiedKFold(
    n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
)
