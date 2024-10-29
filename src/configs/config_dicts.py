from configs.config_enum import (
    ML_MODEL_NAMES,
    OPTIMIZATION_METHODS,
    PROBLEM_TYPES,
    SLLMBO_METHODS,
)


from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from configs.config_params import RANDOM_STATE
from optimizers.fully_llm_with_intelligent_summary import (
    optimize_fully_llm_with_intelligent_summary,
)
from optimizers.fully_llm_with_langchain import optimize_fully_llm_with_langchain
from optimizers.hyperopt_optimizer import optimize_hyperopt
from optimizers.optuna_optimizer import optimize_optuna


ESTIMATORS_DICT = {
    ML_MODEL_NAMES.lightgbm.name: {
        PROBLEM_TYPES.classification.name: LGBMClassifier,
        PROBLEM_TYPES.regression.name: LGBMRegressor,
    },
    ML_MODEL_NAMES.xgboost.name: {
        PROBLEM_TYPES.classification.name: XGBClassifier,
        PROBLEM_TYPES.regression.name: XGBRegressor,
    },
}
DEFAULT_PARAMS_DICT = {
    ML_MODEL_NAMES.lightgbm.name: {
        "verbose": -1,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    ML_MODEL_NAMES.xgboost.name: {
        "verbosity": 0,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
}
OPTIMIZERS_DICT = {
    OPTIMIZATION_METHODS.optuna.name: optimize_optuna,
    OPTIMIZATION_METHODS.hyperopt.name: optimize_hyperopt,
    OPTIMIZATION_METHODS.sllmbo.name: {
        SLLMBO_METHODS.sllmbo_fully_llm_with_intelligent_summary.name: optimize_fully_llm_with_intelligent_summary,  # noqa: E501
        SLLMBO_METHODS.sllmbo_fully_llm_with_langchain.name: optimize_fully_llm_with_langchain,
        SLLMBO_METHODS.sllmbo_llm_tpe.name: optimize_optuna,
    },
}
