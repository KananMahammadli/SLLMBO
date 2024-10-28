# dataset params
DATASET_PATH = r"./src/data"
DATASET_VAR = "dataset"
PROBLEM_TYPE_VAR = "problem_type"
PROBLEM_DESCRIPTION_VAR = "problem_description"
METRIC_VAR = "metric"
METRIC_FUNC_VAR = "metric_func"
DIRECTION_VAR = "direction"
OPTIMIZATION_CHECK_FUNC_VAR = "optmization_check_func"
CV_VAR = "cv"

NEPTUNE_PROJECT_NAME = "kananmaham/LLM-Tuning"  # for neptune expriment tracking

RANDOM_STATE = 42
# general parameters
N_SPLITS = 5
N_TRIALS = 2
N_SUMMARIZE_ITER = 10
MAX_N_ITERS_WITHOUT_IMPROVEMENT = 15
SAMPLE_SIZE = 20000  # sample size to keep of dataset is too large
