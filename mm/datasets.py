import numpy as np
import pandas as pd

def generate_data(n=100):
    n_hospitals = 5

    np.random.seed(123)

    # observed covariates
    treatment = (np.random.random(n) < 0.5).astype(int)
    hospital = np.random.choice(np.arange(n_hospitals), n).astype(int)

    # unobserved covariates
    noise = np.random.normal(0, 1, n)
    intercept_true = 1

    # effect sizes
    treatment_effect = 1
    hospital_intercept = np.random.normal(0, 1, n_hospitals)

    # realized effects
    treatment_realized = np.where(treatment, treatment_effect, 0)
    hospital_realized = hospital_intercept[hospital]

    outcome = intercept_true + treatment_realized + hospital_realized + noise

    data = pd.DataFrame({
        'treatment': treatment,
        'hospital': hospital,
        'outcome': outcome
    })

    return data
