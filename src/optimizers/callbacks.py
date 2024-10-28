import optuna


class EarlyStoppingCallback:
    def __init__(self, patience=None):
        self.patience = patience
        self.best_value = None
        self.n_iters_without_improvement = 0

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if self.patience is None:
            pass  # Early stopping is disabled
        if trial.number > 0:
            current_value = trial.value
            if self.best_value is None:
                self.best_value = current_value
                self.n_iters_without_improvement = 0

            elif current_value is None:
                self.n_iters_without_improvement += 1
    
            elif study.direction == optuna.study.StudyDirection.MAXIMIZE:
                if current_value > self.best_value:
                    self.best_value = current_value
                    self.n_iters_without_improvement = 0
                else:
                    self.n_iters_without_improvement += 1
            else:  # MINIMIZE
                if current_value < self.best_value:
                    self.best_value = current_value
                    self.n_iters_without_improvement = 0
                else:
                    self.n_iters_without_improvement += 1

            if self.n_iters_without_improvement >= self.patience:
                study.stop()
        else:
            self.best_value = trial.value
