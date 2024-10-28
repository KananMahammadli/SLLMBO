from google.api_core.exceptions import ResourceExhausted
from functools import wraps
import time

from configs.config_enum import ML_MODEL_NAMES


def get_param_space(model_name):
    if model_name == ML_MODEL_NAMES.lightgbm.name:
        return {
            "num_leaves": (20, 3000),
            "max_depth": (3, 12),
            "learning_rate": (0.01, 0.3),
            "n_estimators": (100, 1000),
            "min_child_samples": (1, 300),
            "subsample": (0.7, 1.0),
            "colsample_bytree": (0.7, 1.0),
            "reg_alpha": (0, 5),
            "reg_lambda": (0, 5),
        }
    elif model_name == ML_MODEL_NAMES.xgboost.name:
        return {
            "max_depth": (3, 12),
            "learning_rate": (0.01, 0.3),
            "n_estimators": (100, 1000),
            "min_child_weight": (1, 10),
            "subsample": (0.7, 1.0),
            "colsample_bytree": (0.7, 1.0),
            "gamma": (0, 5),
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def api_rate_limiter(max_retries=5, base_delay=1, max_delay=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except ResourceExhausted as e:
                    if attempt == max_retries - 1:
                        print(f"Max retries reached. Error: {e}")
                        return None  # Return None instead of raising to allow fallback
                    delay = min(base_delay * (2**attempt), max_delay)
                    print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
            return None

        return wrapper

    return decorator
