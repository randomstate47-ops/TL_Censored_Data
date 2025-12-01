import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gekko import GEKKO
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

#----- Syhtetic Data Generation -----
np.random.seed(42)
n_samples = 20

x0 = np.random.randn(n_samples)
x1 = np.random.randn(n_samples)
x2 = x0 + 0.5 * x1 
x3 = np.random.randn(n_samples)
x4 = x3 + 2 * x0 - x2
x5 = x4 * x1
x6 = np.random.randn(n_samples)
x7 = x1 + 0.3 * x6
x8 = x3 - x4
x9 = x2 * x6

data = pd.DataFrame({
    'x0': x0, 'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
    'x5': x5, 'x6': x6, 'x7': x7, 'x8': x8, 'x9': x9
})

true_β = np.zeros(10)
true_β[[0,1,2,3,4,5,6,7,8,9]] = [0.333,0.001,4.2,2.4,3,0.6,1,0.0005,0.001,0.001]

# Gaussian noise with standard deviation 0.5.
y = data.dot(true_β) + np.random.randn(n_samples)*0.5

data['True_y'] = y

#----- Inequality Constrained Substitution -----
data['y'] = data['True_y']*1
data.at[1,'y'] = '<0'
data.at[3,'y'] = '>4'
data.at[6,'y'] = '>6'
data.at[0,'y'] = '<10'
data.at[8,'y'] = '<-1'
data.at[10,'y'] = '<0'
data.at[13,'y'] = '<-5'
data.at[15,'y'] = '<0'
data.at[16,'y'] = '<-10'
data.at[19,'y'] = '<-6.5'

#----- Data Engineering -----
#   Each observation is parsed, and two aligned outputs are generated: 
#   an effective measured value (y_meas) and a categorical (bound type) 
#   specifying which constraint will be applied. 

df = data.drop(columns='True_y')
dr = df.copy()
d = df.copy()
d

y_meas, bound_type = [], []
for val in d['y']:
    if isinstance(val, str) and '<' in val:
        y_meas.append(float(val.strip('<')))
        bound_type.append('upper_only')
    elif isinstance(val, str) and '>' in val:
        y_meas.append(float(val.strip('>')))
        bound_type.append('lower_only')
    else:
        y_meas.append(float(val))
        bound_type.append('both')

d['y_meas'] = y_meas
d['bound'] = bound_type

#---- GEKKO Modeling -----

d_train = d.iloc[:16,:]
d_test = d.iloc[16:,:]
X_test = d.iloc[16:,:10]
Y_test = data.iloc[16:,10:11]

n_features = 10 
d = df.copy()
X = d_train[[f'x{i}' for i in range(10)]].values
n_samples, n_features = X.shape

m = GEKKO(remote=False)

# Create one coefficient per feature (FV = Fixed Variable)
β = [m.FV(value=1, name=f'b{i}') for i in range(n_features)]
for b in β:
    b.STATUS = 1
# Slack variables
eU = m.Array(m.Var, n_samples, lb=0)
eL = m.Array(m.Var, n_samples, lb=0)
# Weighting
w_err = 1 / np.std(y_meas)
for i in range(n_samples):
    y_pred = m.Intermediate(sum(β[j] * X[i, j] for j in range(n_features)))
    
    if bound_type[i] == 'upper_only':
        m.Equation(eU[i] >= y_pred - y_meas[i])
        m.Equation(eL[i] == 0)
    elif bound_type[i] == 'lower_only':
        m.Equation(eL[i] >= y_meas[i] - y_pred)
        m.Equation(eU[i] == 0)
    else:  # 'both'
        m.Equation(eU[i] >= y_pred - y_meas[i])
        m.Equation(eL[i] >= y_meas[i] - y_pred)

# Objective: minimize L1 error (weighted)
m.Minimize(w_err * (m.sum(eU) + m.sum(eL)))

# Solver options
m.options.IMODE = 3  # Steady state optimization
m.solve(disp=False)

β_values = [b.value[0] for b in β]

# Predictions
y_predicted = X_test.dot(np.array(β_values))

# Metrics
mse_l1 = mean_squared_error(Y_test, y_predicted)
r2_l1 = r2_score(Y_test, y_predicted)
MAE_l1 = mean_absolute_error(Y_test, y_predicted)

# Results DataFrame
results = pd.DataFrame(X_test, columns=[f'x{i}' for i in range(n_features)])
results['Prediction_y'] = y_predicted
results['True_y'] = Y_test

#----- Traditional Linear Model -----
d_train_LM = d_train[d_train['bound']=='both']
x_lm = d_train_LM.iloc[:,:10]
y_lm = d_train_LM.iloc[:,11:12]

x_train_lm = x_lm.values
y_train_lm = y_lm.values

# Same test as L1-Norm
#X_test
#Y_test

model = LinearRegression().fit(x_train_lm, y_train_lm)
y_pred_train_lm = model.predict(x_train_lm)
y_pred_test_lm = model.predict(X_test)

r2_lm = r2_score(Y_test, y_pred_test_lm)
MSE_lm = mean_squared_error(Y_test, y_pred_test_lm)
MAE_lm = mean_absolute_error(Y_test, y_pred_test_lm)

#----- Linear Model ---- BOOTSTRAPPING for UQ -----
def bootstrap_train_set_lm(x_train_lm, y_train_lm, random_state=10):
    rng = np.random.default_rng(random_state)
    n_samples = x_train_lm.shape[0]
    indices = rng.choice(n_samples, size=n_samples, replace=True)
    X_boot_lm = x_train_lm[indices]
    y_boot_lm = y_train_lm[indices]
    return X_boot_lm, y_boot_lm

n_bootstrap_lm = 3
X_boot_lm_list = []
y_boot_lm_list = []

for i in range(n_bootstrap_lm):
    X_boot_lm, y_boot_lm = bootstrap_train_set_lm(x_train_lm, 
                                                  y_train_lm, 
                                                  random_state=i)
    X_boot_lm_list.append(X_boot_lm)
    y_boot_lm_list.append(y_boot_lm)

train_preds_lm = []
test_preds_lm = []

from sklearn.linear_model import LinearRegression
lr = LinearRegression

for b in range(n_bootstrap_lm):
    Xb = X_boot_lm_list[b]
    yb = y_boot_lm_list[b].ravel()

    model = lr().fit(Xb, yb)
    train_pred_lm = model.predict(Xb)
    train_preds_lm.append(train_pred_lm)

    test_pred_lm = model.predict(X_test)
    test_preds_lm.append(test_pred_lm)

test_preds_lm = np.stack(test_preds_lm, axis=0)
boot_mean    = test_preds_lm.mean(axis=0)
boot_std     = test_preds_lm.std(axis=0, ddof=1)

per_model_metrics_lm = []
for b in range(n_bootstrap_lm):
    r2  = r2_score(Y_test, test_preds_lm[b])
    mse = mean_squared_error(Y_test, test_preds_lm[b])
    mae = mean_absolute_error(Y_test, test_preds_lm[b])
    per_model_metrics_lm.append((r2, mse, mae))

r2_boot_lm  = r2_score(Y_test, boot_mean)
mse_boot_lm = mean_squared_error(Y_test, boot_mean)
mae_boot_lm = mean_absolute_error(Y_test, boot_mean)

#----- L1-NORM ---- BOOTSTRAPPING for UQ -----
X_t = d_train.iloc[:,:10].values
y_t = d_train['y'].values
y_m = d_train['y_meas'].values
bound_t = d_train['bound'].values
X_test = X_test.values
Y_test = Y_test.values

def bootstrap_train_set(X_t, y_t, y_m, bound_t, random_state=10):
    rng = np.random.default_rng(random_state)
    n_samples = X_t.shape[0]
    indices = rng.choice(n_samples, size=n_samples, replace=True)
    X_boot = X_t[indices]
    y_boot = y_t[indices]
    ym_boot = y_m[indices]
    bound_boot = bound_t[indices]
    return X_boot, y_boot, ym_boot, bound_boot

n_bootstrap = 3
rng = np.random.default_rng(123)

X_boot_list = []
y_boot_list = []
ym_boot_list = []
bound_boot_list = []

for i in range(n_bootstrap):
    X_boot, y_boot, ym_boot, bound_boot = bootstrap_train_set(X_t, 
                                                              y_t, 
                                                              y_m, 
                                                              bound_t, 
                                                              random_state=i)
    X_boot_list.append(X_boot)
    y_boot_list.append(y_boot)
    ym_boot_list.append(ym_boot)
    bound_boot_list.append(bound_boot)

# --- GEKKO  for one bootstrap sample ---
def train_gekko_linear(X_boot, y_meas_boot, bound_boot):

    m = GEKKO(remote=False)

    n_samples, n_features_ = X_boot.shape
    assert n_features_ == n_features, "Feature mismatch"
    beta = [m.FV(value=0.0, name=f'b{j}') for j in range(n_features)]
    for b in beta:
        b.STATUS = 1

    eU = m.Array(m.Var, n_samples, lb=0)
    eL = m.Array(m.Var, n_samples, lb=0)

    std_y = float(np.std(y_meas_boot)) if np.std(y_meas_boot) > 0 else 1.0
    w_err = 1.0 / std_y
    for i in range(n_samples):
        y_pred_i = m.Intermediate(m.sum([beta[j] * X_boot[i, j] for j in range(n_features)]))
        if bound_boot[i] == 'upper_only':
            m.Equation(eU[i] >= y_pred_i - y_meas_boot[i])
            m.Equation(eL[i] == 0)
        elif bound_boot[i] == 'lower_only':
            m.Equation(eL[i] >= y_meas_boot[i] - y_pred_i)
            m.Equation(eU[i] == 0)
        else:  # 'both'
            m.Equation(eU[i] >= y_pred_i - y_meas_boot[i])
            m.Equation(eL[i] >= y_meas_boot[i] - y_pred_i)

    m.Minimize(w_err * (m.sum(eU) + m.sum(eL)))

    # Solve
    m.options.IMODE = 3 
    m.solve(disp=False)
    beta_vals = np.array([b.value[0] for b in beta])
    return beta_vals

# Bootstrapping, fit models, and prediction on test ---
betas = []
train_preds = []
test_preds = []

for b in range(n_bootstrap):
    X_boot, y_boot, y_meas_boot, bound_boot = bootstrap_train_set(X_t, y_t, y_m, bound_t, rng)
    beta_b = train_gekko_linear(X_boot, y_meas_boot, bound_boot)
    betas.append(beta_b)

    y_pred_b = X_test.dot(beta_b).reshape(-1, 1)
    y_train_preds = X_t.dot(beta_b).reshape(-1, 1)
    train_preds.append(y_train_preds)
    test_preds.append(y_pred_b)

betas       = np.stack(betas, axis=0)
test_preds  = np.stack(test_preds, axis=0)
ens_mean    = test_preds.mean(axis=0)
ens_std     = test_preds.std(axis=0, ddof=1)

# --- 4) Metrics: per-model + ensemble mean prediction ---
per_model_metrics = []
for b in range(n_bootstrap):
    r2  = r2_score(Y_test, test_preds[b])
    mse = mean_squared_error(Y_test, test_preds[b])
    mae = mean_absolute_error(Y_test, test_preds[b])
    per_model_metrics.append((r2, mse, mae))

r2_ens  = r2_score(Y_test, ens_mean)
mse_ens = mean_squared_error(Y_test, ens_mean)
mae_ens = mean_absolute_error(Y_test, ens_mean)

# --- 5) Results DataFrame ---
results = pd.DataFrame(X_test, columns=[f'x{i}' for i in range(n_features)])
results['y_true']     = Y_test.reshape(-1)
results['y_pred_mean']= ens_mean.reshape(-1)
results['y_pred_std'] = ens_std.reshape(-1)

# 95% normal-approx conf band for the model mean (not a PI)
z = 1.96
results['mean_lo_95'] = results['y_pred_mean'] - z * results['y_pred_std']
results['mean_hi_95'] = results['y_pred_mean'] + z * results['y_pred_std']

# ----- Traditional Linear Model ---- Custom Metrics -----
def _to_1d_array(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "cpu") and callable(x.cpu):
        x = x.cpu().numpy()
    x = np.asarray(x).squeeze().ravel()
    return x

def evaluate_models(test_preds_lm, d_test, Y_test, models=(0,1,2)):
    base = pd.DataFrame({
        'y'      : d_test['y'].to_numpy(),
        'y_meas' : d_test['y_meas'].to_numpy(),
        'True_y' : _to_1d_array(Y_test),
        'bound'  : d_test['bound'].astype(str).str.lower().str.strip().to_numpy()
    }, index=d_test.index)

    metrics = []
    long_rows = []

    for i in models:
        pred_i = _to_1d_array(test_preds_lm[i])
        if len(pred_i) != len(base):
            raise ValueError(f"Length mismatch for model {i}: pred={len(pred_i)} vs base={len(base)}")

        r = base.copy()
        r[f'pred{i}'] = pred_i
        # Flags
        r['upper_only_flag'] = ((r['bound'] == 'upper_only') & (r[f'pred{i}'] < r['y_meas'])).astype(int)
        r['lower_only_flag'] = ((r['bound'] == 'lower_only') & (r[f'pred{i}'] > r['y_meas'])).astype(int)
        r['both_flag']       = (r['bound'] == 'both').astype(int)
        # Violations
        delta_upper = np.where(r['bound'] == 'upper_only',
                               np.maximum(r[f'pred{i}'] - r['y_meas'], 0),
                               0)
        delta_lower = np.where(r['bound'] == 'lower_only',
                               np.maximum(r['y_meas'] - r[f'pred{i}'], 0),
                               0)
        # Total inequality violation; for 'both' use |True_y - y_meas|
        r['Delta'] = (
            delta_upper
            + delta_lower
            + np.where(r['bound'] == 'both', np.abs(r['True_y'] - r['y_meas']), 0)
        )
        # Metrics
        bb = r['both_flag'].sum()
        tc = len(r)
        ab = tc - bb  # number of inequality-only rows
        kb = r['upper_only_flag'].sum() + r['lower_only_flag'].sum()

        A = kb / ab if ab > 0 else np.nan
        MACV = r['Delta'].sum() / ab if ab > 0 else np.nan

        metrics.append({'model': i, 'A': A, 'MACV': MACV, 'n_total': tc, 'n_both': bb, 'n_ineq': ab})
        r['model'] = i

        # Keep a tidy/long selection; add others if you like
        long_rows.append(
            r[['model', 'y', 'y_meas', 'True_y', f'pred{i}', 'bound',
               'upper_only_flag', 'lower_only_flag', 'both_flag', 'Delta']].rename(
                columns={f'pred{i}': 'pred'})
        )

    metrics_df_lm = pd.DataFrame(metrics).sort_values('model').reset_index(drop=True)
    long_df_lm = pd.concat(long_rows, axis=0, ignore_index=True)

    return metrics_df_lm, long_df_lm

metrics_df_lm, all_models_long_lm = evaluate_models(test_preds_lm, d_test, Y_test, models=(0,1,2))

# ----- L1-Norm ---- Custom Metrics -----
r0 = pd.DataFrame({'y':d_test['y'], 'y_meas':d_test['y_meas'],
                   'pred0':test_preds[0].squeeze(),
                   'True_y':Y_test.squeeze(),
                   'bound':d_test['bound']})
r0['bound'] = r0['bound'].astype(str).str.lower().str.strip()

r0['upper_only_flag'] = ((r0['bound'] == 'upper_only') & (r0['pred0'] < r0['y_meas'])).astype(int)
r0['lower_only_flag'] = ((r0['bound'] == 'lower_only') & (r0['pred0'] > r0['y_meas'])).astype(int)
r0['both_flag']       = (r0['bound'] == 'both').astype(int)

delta_upper = np.where(r0['bound'] == 'upper_only',
                       np.maximum(r0['pred0'] - r0['y_meas'], 0),
                       0)

delta_lower = np.where(r0['bound'] == 'lower_only',
                       np.maximum(r0['y_meas'] - r0['pred0'], 0),
                       0)

r0['Delta'] = (
    delta_upper
    + delta_lower
    + np.where(r0['bound'] == 'both', (r0['True_y'] - r0['y_meas']).abs(), 0)
)

bb = r0['both_flag'].sum()
tc = r0['both_flag'].count()
ab = tc-bb
kb = r0['upper_only_flag'].sum() + r0['lower_only_flag'].sum()

A = kb/ab
MACV = r0['Delta'].sum()/ab


def _to_1d_array(x):
    """Safely convert list/np/torch tensors to 1-D numpy arrays."""
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    elif hasattr(x, "cpu") and callable(x.cpu):  # other torch-like
        x = x.cpu().numpy()
    x = np.asarray(x).squeeze().ravel()
    return x

def evaluate_models(test_preds, d_test, Y_test, models=(0,1,2)):
    # --- Base frame (once) ---
    base = pd.DataFrame({
        'y'      : d_test['y'].to_numpy(),
        'y_meas' : d_test['y_meas'].to_numpy(),
        'True_y' : _to_1d_array(Y_test),
        'bound'  : d_test['bound'].astype(str).str.lower().str.strip().to_numpy()
    }, index=d_test.index)

    metrics = []
    long_rows = []

    for i in models:
        pred_i = _to_1d_array(test_preds[i])
        if len(pred_i) != len(base):
            raise ValueError(f"Length mismatch for model {i}: pred={len(pred_i)} vs base={len(base)}")
        r = base.copy()
        r[f'pred{i}'] = pred_i
        # Flags
        r['upper_only_flag'] = ((r['bound'] == 'upper_only') & (r[f'pred{i}'] < r['y_meas'])).astype(int)
        r['lower_only_flag'] = ((r['bound'] == 'lower_only') & (r[f'pred{i}'] > r['y_meas'])).astype(int)
        r['both_flag']       = (r['bound'] == 'both').astype(int)
        # Violations
        delta_upper = np.where(r['bound'] == 'upper_only',
                               np.maximum(r[f'pred{i}'] - r['y_meas'], 0),
                               0)
        delta_lower = np.where(r['bound'] == 'lower_only',
                               np.maximum(r['y_meas'] - r[f'pred{i}'], 0),
                               0)
        # Total inequality violation; for 'both' use |True_y - y_meas|
        r['Delta'] = (
            delta_upper
            + delta_lower
            + np.where(r['bound'] == 'both', np.abs(r['True_y'] - r['y_meas']), 0)
        )

        # Metrics
        bb = r['both_flag'].sum()
        tc = len(r)
        ab = tc - bb  # number of inequality-only rows
        kb = r['upper_only_flag'].sum() + r['lower_only_flag'].sum()

        A = kb / ab if ab > 0 else np.nan
        MACV = r['Delta'].sum() / ab if ab > 0 else np.nan

        metrics.append({'model': i, 'A': A, 'MACV': MACV, 'n_total': tc, 'n_both': bb, 'n_ineq': ab})
        r['model'] = i
        long_rows.append(
            r[['model', 'y', 'y_meas', 'True_y', f'pred{i}', 'bound',
               'upper_only_flag', 'lower_only_flag', 'both_flag', 'Delta']].rename(
                columns={f'pred{i}': 'pred'})
        )

    metrics_df = pd.DataFrame(metrics).sort_values('model').reset_index(drop=True)
    long_df = pd.concat(long_rows, axis=0, ignore_index=True)

    return metrics_df, long_df

metrics_df, all_models_long = evaluate_models(test_preds, d_test, Y_test, models=(0,1,2))

r1 = pd.DataFrame({'y':d_test['y'], 'y_meas':d_test['y_meas'],'pred0':test_preds[1].squeeze(),
                   'True_y':Y_test.squeeze(),'bound':d_test['bound']})

r1['bound'] = r1['bound'].astype(str).str.lower().str.strip()

# Flags
r1['upper_only_flag'] = ((r1['bound'] == 'upper_only') & (r1['pred0'] < r1['y_meas'])).astype(int)
r1['lower_only_flag'] = ((r1['bound'] == 'lower_only') & (r1['pred0'] > r1['y_meas'])).astype(int)
r1['both_flag']       = (r1['bound'] == 'both').astype(int)

# Violation magnitudes:
# - upper_only: violation if pred > y  → amount = pred - y
# - lower_only: violation if pred < y  → amount = y - pred
delta_upper = np.where(r1['bound'] == 'upper_only',
                       np.maximum(r1['pred0'] - r1['y_meas'], 0),
                       0)

delta_lower = np.where(r0['bound'] == 'lower_only',
                       np.maximum(r0['y_meas'] - r1['pred0'], 0),
                       0)

r1['Delta'] = (
    delta_upper
    + delta_lower
    + np.where(r0['bound'] == 'both', (r1['True_y'] - r1['y_meas']).abs(), 0)
)

bb = r1['both_flag'].sum()
tc = r1['both_flag'].count()
ab = tc-bb
kb = r1['upper_only_flag'].sum() + r1['lower_only_flag'].sum()

A = kb/ab
MACV = r1['Delta'].sum()/ab


from scipy.stats import t
n_models  = test_preds.shape[0]
n_test = test_preds.shape[1]

alpha = 0.05
t_crit = t.ppf(1 - alpha/2, df=n_models - 1)

# 1. SUCI (for the mean) - L1_NORM
means = np.mean(test_preds, axis=0)
stds = np.std(test_preds, axis=0, ddof=1)
SEs = stds / np.sqrt(n_models)
suci_half_width = t_crit * SEs

# 1. SUCI (for the mean) - LM
means_lm = np.mean(test_preds_lm, axis=0)
stds_lm = np.std(test_preds_lm, axis=0, ddof=1)
SEs_lm = stds_lm / np.sqrt(n_models)
suci_half_width_lm = t_crit * SEs_lm

# --- 2. Estimate data noise variance from training data ---
True_y_train = d.copy()
True_y_train['True_y'] = data['True_y']

True_y_train = True_y_train.iloc[:16,:]

mean_pred_train = np.mean(train_preds, axis=0)   # (n_train,)
residuals = True_y_train['True_y'].values - mean_pred_train                     # (n_train,)
s_squared = np.var(residuals, ddof=1)                     # scalar

# --- 3. PI for a new observation at each test point ---
# This includes both model (ensemble) variance and data noise variance
pi_half_width = t_crit * np.sqrt(SEs**2 + s_squared)      # (n_test,)

# --- 4. Get lower and upper bounds for intervals ---
suci_lowers = means - suci_half_width
suci_uppers = means + suci_half_width

suci_lowers_lm = means_lm - suci_half_width_lm
suci_uppers = means_lm + suci_half_width_lm

pi_lowers   = means - pi_half_width
pi_uppers   = means + pi_half_width

e1 =test_preds[0]
e2 =test_preds[1]
e3 =test_preds[2]

e1_lm =test_preds_lm[0]
e2_lm =test_preds_lm[1]
e3_lm =test_preds_lm[2]

# --- 5. Plot ---
x_axis = np.arange(n_test)

plt.plot(figsize=(7, 4.5))


plt.errorbar(x_axis, means.ravel(), yerr=suci_half_width.ravel(), fmt='o',
             alpha=0.6, 
             capsize=4,
             linewidth=2, 
             label='Mean CI-L1NO', color='red')

plt.errorbar(x_axis, means_lm.ravel(), yerr=suci_half_width_lm.ravel(), fmt='o',
             alpha=0.6,
             capsize=4, 
             label='Mean CI-Linear Model', color='black')

plt.plot(x_axis, e1, 'xr', alpha=0.7, label='L1-NORM Bootstrapping Predictions')
plt.plot(x_axis, e2, 'xr', alpha=0.7)
plt.plot(x_axis, e3, 'xr', alpha=0.7)


plt.plot(x_axis, e1_lm, 'xk', alpha=0.7, label='Linear Model Bootstrapping Predictions')
plt.plot(x_axis, e2_lm, 'xk', alpha=0.7)
plt.plot(x_axis, e3_lm, 'xk', alpha=0.7)


plt.plot(x_axis, Y_test, 'dm', alpha=0.7, label='True_y')


Y_test_1d = np.asarray(Y_test).ravel()
for i, value in enumerate(Y_test_1d):
    plt.text(i, value + 0.5, f'{value:.2f}', rotation=90, ha='right', va='baseline')


print('TRADITIONAL LINEAR MODEL')
print('Performance Metrics:')
print("R²  :", r2_score(Y_test, y_pred_test_lm))
print("MSE :", mean_squared_error(Y_test, y_pred_test_lm))
print("MAE :", mean_absolute_error(Y_test, y_pred_test_lm))
print('')
print(f"Traditional Linear Model - Bootstrapping (mean prediction)")
print(f'R²: {r2_boot_lm:.4f} | MSE: {mse_boot_lm:.4f} | MAE: {mae_boot_lm:.4f}')
print("Per-model metrics:")
for i, (r2_i, mse_i, mae_i) in enumerate(per_model_metrics_lm[:], 1):
    print(f"  Model {i:02d}: R²={r2_i:.4f} | MSE={mse_i:.4f} | MAE={mae_i:.4f}")
print('')
print('Traditional Linear Model Constrained Violation Metrics')
print(metrics_df_lm)

print('')
print('')

print('L1-NORM MODEL')
print('Performance Metrics:')
print(f'R²: {r2_l1:.4f}')
print(f'MSE: {mse_l1:.4f}')
print(f'MAE: {MAE_l1:.4f}')
print('')
print(f"L1-NORM Model - Bootstrapping (mean prediction)")
print(f'R²: {r2_ens:.4f} | MSE: {mse_ens:.4f} | MAE: {mae_ens:.4f}')
print("Per-model metrics:")
for i, (r2_i, mse_i, mae_i) in enumerate(per_model_metrics[:], 1):
    print(f"  Model {i:02d}: R²={r2_i:.4f} | MSE={mse_i:.4f} | MAE={mae_i:.4f}")
print('')
print('L1-Norm Constrained Violation Metrics')
print(metrics_df)

plt.xlabel('Test Point Index')
plt.ylabel('Bootstrapping Prediction')
plt.title('95% Confidence Intervals')
plt.legend()
plt.tight_layout()
#plt.savefig("Boot_SUCI.png", format="png", dpi=300)
plt.show()


