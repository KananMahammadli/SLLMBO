import optuna
from sklearn.model_selection import cross_val_score
from optimizers.callbacks import EarlyStoppingCallback
from optimizers.util import get_param_space


def optimize_optuna(
    X,
    y,
    model_name,
    estimator,
    cv,
    study_name,
    storage,
    default_params={},
    n_trials=50,
    metric="f1_weighted",
    direction="maximize",
    random_state=42,
    sampler=None,
    patience=None,
):
    def objective(trial):
        params = {}
        param_space = get_param_space(model_name)

        for param, (low, high) in param_space.items():
            if param in [
                "num_leaves",
                "max_depth",
                "n_estimators",
                "min_child_samples",
            ]:
                params[param] = trial.suggest_int(param, low, high)
            else:
                params[param] = trial.suggest_float(param, low, high)
        all_params = {**params, **default_params}
        model = estimator(**all_params)
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
        if scores.mean() < 0:
            scores = -scores
        return scores.mean()

    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=random_state)
    if patience is not None and patience > 0:
        callbacks = [EarlyStoppingCallback(patience=patience)]
    else:
        callbacks = None
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
        sampler=sampler,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=callbacks,
    )

    best_params = study.best_params
    best_score = study.best_value

    return best_params, best_score, study
