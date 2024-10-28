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

ADULT_CENSUS_METADATA = {}

path_adult_census = f"{DATASET_PATH}/{DATASET_NAMES.adult_census.name}.csv"

# prepare dataset
X_train, X_test, y_train, y_test = load_data(
    file_path=path_adult_census,
    target_column="class",
    drop_columns=[
        "id",
        "ID",
        "native-country",
        "fnlwgt",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
    ],
    shuffle=True,
    random_state=RANDOM_STATE,
)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
ADULT_CENSUS_METADATA[DATASET_VAR] = (
    X_train,
    X_test,
    y_train,
    y_test,
)
ADULT_CENSUS_METADATA[PROBLEM_TYPE_VAR] = PROBLEM_TYPES.classification.name

# problem description
ADULT_CENSUS_METADATA[PROBLEM_DESCRIPTION_VAR] = PROBLEM_DESCRIPTIONS[
    DATASET_NAMES.adult_census.name
]

# optimization parameters
ADULT_CENSUS_METADATA[METRIC_VAR] = METRICS.f1_weighted.name
ADULT_CENSUS_METADATA[METRIC_FUNC_VAR] = lambda y_true, y_pred: f1_score(
    y_true, y_pred, average="weighted"
)
ADULT_CENSUS_METADATA[DIRECTION_VAR] = OPTIMIZATION_DIRECTIONS.maximize.name
ADULT_CENSUS_METADATA[OPTIMIZATION_CHECK_FUNC_VAR] = max
ADULT_CENSUS_METADATA[CV_VAR] = StratifiedKFold(
    n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
)
