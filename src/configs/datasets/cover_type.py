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
    SAMPLE_SIZE,
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

COVER_TYPE_METADATA = {}

path_cover_type = f"{DATASET_PATH}/{DATASET_NAMES.cover_type.name}.csv"

# prepare dataset
X_train, X_test, y_train, y_test = load_data(
    file_path=path_cover_type,
    target_column="Cover_Type",
    drop_columns=["Id"],
    shuffle=True,
    random_state=RANDOM_STATE,
    sample_size=SAMPLE_SIZE,
)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
COVER_TYPE_METADATA[DATASET_VAR] = (
    X_train.iloc[:, :5],
    X_test.iloc[:, :5],
    y_train,
    y_test,
)
COVER_TYPE_METADATA[PROBLEM_TYPE_VAR] = PROBLEM_TYPES.classification.name

# problem description
COVER_TYPE_METADATA[PROBLEM_DESCRIPTION_VAR] = PROBLEM_DESCRIPTIONS[
    DATASET_NAMES.cover_type.name
]

# optimization parameters
COVER_TYPE_METADATA[METRIC_VAR] = METRICS.f1_weighted.name
COVER_TYPE_METADATA[METRIC_FUNC_VAR] = lambda y_true, y_pred: f1_score(
    y_true, y_pred, average="weighted"
)
COVER_TYPE_METADATA[DIRECTION_VAR] = OPTIMIZATION_DIRECTIONS.maximize.name
COVER_TYPE_METADATA[OPTIMIZATION_CHECK_FUNC_VAR] = max
COVER_TYPE_METADATA[CV_VAR] = StratifiedKFold(
    n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
)
