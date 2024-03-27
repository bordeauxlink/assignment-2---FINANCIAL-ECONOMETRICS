
#a
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

def calculate_ic(df, maxlags):
    # Convert all columns to float
    df = df.astype(float)
    
    ic_values = []

    for lag in maxlags:
        model = VAR(df)
        results = model.fit(lag)
        ic_values.append([lag, results.aic, results.bic, results.hqic])

    ic_df = pd.DataFrame(ic_values, columns=['Lag', 'AIC', 'BIC', 'HQIC'])
    return ic_df

# Load data
df = pd.read_csv('PJM.csv', parse_dates=['Date'], index_col='Date')
df.columns = ['Zonal_COMED_price', 'System_load_forecast', 'Zonal_COMED_load_forecast']
df = df.astype(float)

maxlags = [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
ic_df = calculate_ic(df, maxlags)
print(ic_df)

#b
from statsmodels.tsa.vector_ar.var_model import VARResults
import pandas as pd
def granger_causality_tests(res: VARResults, lag: int) -> pd.DataFrame:
    variables = res.model.endog_names
    f_stats = pd.DataFrame(index=variables, columns=variables)

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            test_result = res.test_causality(var1, var2, kind='f', signif=0.05)
            f_stats.loc[var1, var2] = test_result.test_statistic

    return f_stats
model = VAR(df)
lag = ic_df['AIC'].idxmin()
results = model.fit(lag)

f_stats = granger_causality_tests(results, lag)
print(f_stats)

#c
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR

# Fit a VAR(3) model
model = VAR(df)
results = model.fit(3)

# Compute impulse responses
irf = results.irf(12)

# Plot response of Zload to a shock in each variable
for var in ['Zonal_COMED_price', 'System_load_forecast', 'Zonal_COMED_load_forecast']:
    irf.plot(impulse='Zonal_COMED_load_forecast', response=var)
    plt.title(f'Response of Zload to a shock in {var}')
    plt.figure(figsize=(10, 5))
    plt.tight_layout()
    plt.show()