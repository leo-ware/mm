from mm.api import Model
import pandas as pd
import numpy as np

class DesignMatrix:
    def __init__(self, model: Model, data: pd.DataFrame):
        self.model = model
        self.data = data
    
    def outcome(self):
        return self.data[self.model.outcome.name]
    
    def fixed_effects(self):
        columns = [fe.name.name for fe in self.model.effect.fixed_effects]
        return self.data[columns]
    
    def fe_regress(self):
        fe = self.fixed_effects()
        X = fe.values
        fit_intercept = 1 in self.model.effect

        if fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        beta = np.linalg.inv(X.T @ X) @ X.T @ self.outcome().values

        if fit_intercept:
            return pd.Series(beta[1:], index=fe.columns), beta[0]
        else:
            return pd.Series(beta, index=fe.columns), None
    
    def outcome_fe_resid(self):
        beta, intercept = self.fe_regress()
        fe = self.fixed_effects()[beta.index]
        y_hat = fe @ beta.values + (intercept or 0)
        return self.outcome() - y_hat
