import pandas as pd
import numpy as np
import scipy.stats as stats

# -------------------------------------------------------------------
# FASE 1: ESTADÍSTICA DESCRIPTIVA
# -------------------------------------------------------------------

def get_frequency_analysis(df_exploded):
    """
    Returns frequency counts of balls, mean frequency, and standard deviation.
    """
    freqs = df_exploded['Numero'].value_counts().sort_index()
    mean_freq = freqs.mean()
    std_freq = freqs.std()
    
    df_freq = pd.DataFrame({
        'Numero': freqs.index.astype(str),
        'Frecuencia': freqs.values,
        'Media_Esperada': mean_freq
    })
    
    return df_freq, mean_freq, std_freq

def get_sum_distribution(df_draws):
    """
    Returns the distribution of the sum of the balls, and Shapiro-Wilk test.
    """
    df = df_draws.copy()
    df['Suma'] = df['Bolillas_Clean'].apply(lambda x: sum([int(n) for n in x if n.isdigit()]))
    sums = df['Suma']
    
    mean_sum = sums.mean()
    std_sum = sums.std()
    
    # Normalidad: Shapiro-Wilk
    if len(sums) >= 3:
        stat, p_value = stats.shapiro(sums)
    else:
        stat, p_value = 0, 1.0
        
    return sums, mean_sum, std_sum, p_value

def get_parity_analysis(df_draws):
    """
    Returns parity distribution and basic hypergeometric probabilities.
    """
    def count_parity(bolillas):
        nums = [int(n) for n in bolillas if n.isdigit()]
        evens = sum(1 for n in nums if n % 2 == 0)
        odds = len(nums) - evens
        return f"{evens}P-{odds}I"

    parity_counts = df_draws['Bolillas_Clean'].apply(count_parity).value_counts().reset_index()
    parity_counts.columns = ['Combinacion', 'FrecuenciaObservada']
    parity_counts['ProporcionObservada'] = parity_counts['FrecuenciaObservada'] / len(df_draws)
    
    # Calculate Theoretical Probabilities (M=50, n=25 odds, N=6 draws typical)
    # Note: Tinka uses 6 balls from 50 (ignoring boliyapa for base game here).
    def expected_prob(comb_str):
        try:
            parts = comb_str.split('-')
            odds = int(parts[1].replace('I', ''))
            return stats.hypergeom.pmf(odds, 50, 25, 6)
        except:
            return 0
        
    parity_counts['ProbabilidadTeorica'] = parity_counts['Combinacion'].apply(expected_prob)
    
    return parity_counts

# -------------------------------------------------------------------
# FASE 2: ESTADÍSTICA INFERENCIAL
# -------------------------------------------------------------------

def get_chi_square_test(df_exploded, total_balls=50):
    """
    Chi-Square Goodness of Fit test for uniform distribution.
    """
    observed_freq = df_exploded['Numero'].value_counts().sort_index()
    
    total_draws_balls = observed_freq.sum()
    expected_freq = np.full(total_balls, total_draws_balls / total_balls)
    
    # Ensure observed_freq covers all 1 to 50
    obs_all = []
    for i in range(1, total_balls + 1):
        if i in observed_freq.index:
            obs_all.append(observed_freq[i])
        else:
            obs_all.append(0)
            
    chi2_stat, p_value = stats.chisquare(f_obs=obs_all, f_exp=expected_freq)
    
    return chi2_stat, p_value, obs_all, expected_freq

def get_gap_metrics(df_exploded, current_sorteo_max):
    """
    Z-Score Gap Map: Measures how many standard deviations a number is from its expected return.
    """
    gaps = {}
    for num in range(1, 51):
        draws_with_num = df_exploded[df_exploded['Numero'] == num]['Sorteo'].astype(str).str.extract(r'(\d+)').astype(float).squeeze().sort_values(ascending=False).values
        
        if len(draws_with_num) > 1:
            diffs = np.abs(np.diff(draws_with_num))
            mean_gap = np.mean(diffs)
            std_gap = np.std(diffs)
            last_seen = draws_with_num[0]
            current_gap = current_sorteo_max - last_seen
            
            z_score = (current_gap - mean_gap) / std_gap if std_gap > 0 else 0
            
            gaps[num] = {
                'Numero': str(num),
                'Mean_Gap': mean_gap,
                'Current_Gap': current_gap,
                'Z_Score': z_score,
                'Plot_Size': max(1, 5 + (z_score * 2)) 
            }
        else:
            gaps[num] = {'Numero': str(num), 'Mean_Gap': 0, 'Current_Gap': 0, 'Z_Score': 0, 'Plot_Size': 1}
            
    df_gaps = pd.DataFrame(gaps).T
    for col in ['Mean_Gap', 'Current_Gap', 'Z_Score', 'Plot_Size']:
        df_gaps[col] = df_gaps[col].astype(float)
        
    anomaly = df_gaps.loc[df_gaps['Z_Score'].idxmax()]
    
    return df_gaps, anomaly

def get_runs_test(df_draws):
    """
    Runs Test (Wald-Wolfowitz) on Sums to check independence.
    """
    df = df_draws.copy()
    df['Suma'] = df['Bolillas_Clean'].apply(lambda x: sum([int(n) for n in x if n.isdigit()]))
    median_sum = df['Suma'].median()
    
    seq = (df['Suma'] > median_sum).astype(int).values
    
    n1 = np.sum(seq)
    n2 = len(seq) - n1
    runs = np.sum(np.abs(np.diff(seq))) + 1
    
    expected_runs = ((2 * n1 * n2) / (n1 + n2)) + 1 if (n1 + n2) > 0 else 0
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1)) if (n1 + n2) > 1 else 0
    
    z_stat = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return z_stat, p_value, runs, expected_runs

# -------------------------------------------------------------------
# FASE 3: MACHINE LEARNING & AI
# -------------------------------------------------------------------

def get_ml_features(df_exploded):
    """
    Feature Engineering: Generate lags, moving averages, etc. for ML.
    """
    df = df_exploded.copy()
    df.sort_values(by=['Sorteo', 'Numero'], inplace=True)
    
    # Simple feature engineering for demo
    # Number of times a ball has appeared up to this point
    df['Sorteos_Number'] = df['Sorteo'].astype(str).str.extract(r'(\d+)').astype(float)
    df['Cumulative_Count'] = df.groupby('Numero').cumcount() + 1
    
    return df

def train_xgb_model(df_exploded):
    """
    Trains a simple XGBClassifier to predict if a number will show up based on engineered features.
    Returns metrics (Confusion Matrix, ROC Curve data).
    """
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix, roc_curve, auc
        
        df = get_ml_features(df_exploded)
        
        # Super simplified dataset generation for binary classification:
        # We need positive (number drawn) and negative (number not drawn) samples.
        sorteos = df['Sorteo'].unique()
        all_combinations = pd.MultiIndex.from_product([sorteos, range(1, 51)], names=['Sorteo', 'Numero']).to_frame(index=False)
        merged = pd.merge(all_combinations, df[['Sorteo', 'Numero', 'Cumulative_Count']], on=['Sorteo', 'Numero'], how='left')
        merged['Is_Drawn'] = merged['Cumulative_Count'].notna().astype(int)
        merged['Cumulative_Count'] = merged.groupby('Numero')['Cumulative_Count'].ffill().fillna(0)
        
        # Add some random lag features to justify ML use
        merged['Lag_1'] = merged.groupby('Numero')['Is_Drawn'].shift(1).fillna(0)
        merged['Lag_2'] = merged.groupby('Numero')['Is_Drawn'].shift(2).fillna(0)
        merged['Rolling_Sum_5'] = merged.groupby('Numero')['Is_Drawn'].transform(lambda x: x.rolling(5).sum()).fillna(0)
        
        X = merged[['Cumulative_Count', 'Lag_1', 'Lag_2', 'Rolling_Sum_5']]
        y = merged['Is_Drawn']
        
        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        # Train
        model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, scale_pos_weight=(len(y)-y.sum())/y.sum())
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        importance = model.feature_importances_
        feature_names = X.columns
        importances = dict(zip(feature_names, importance))
        
        return cm, fpr, tpr, roc_auc, importances
    except ImportError:
        return None, None, None, None, None

def get_lstm_simulated_loss():
    """
    Returns simulated Loss/Epoch values for an LSTM network placeholder.
    """
    epochs = np.arange(1, 51)
    # Simulate a loss curve that converges early and plateaus
    loss = 0.8 * np.exp(-0.1 * epochs) + 0.5 + np.random.normal(0, 0.02, 50)
    val_loss = 0.8 * np.exp(-0.08 * epochs) + 0.6 + np.random.normal(0, 0.03, 50)
    
    return pd.DataFrame({'Epoch': epochs, 'Train_Loss': loss, 'Val_Loss': val_loss})

def get_bayesian_inference(df_exploded, current_sorteo_max):
    """
    Simple Bayesian Evidence Updator.
    Prior: 6/50 (base chance of any number being drawn).
    Likelihood: P(Evidence | Drawn) vs P(Evidence | Not Drawn) based on gap length.
    """
    df_gaps, _ = get_gap_metrics(df_exploded, current_sorteo_max)
    
    # Prior Probability (P(D))
    prior = 6 / 50.0  # ~12%
    
    results = []
    for _, row in df_gaps.iterrows():
        # Heuristic likelihood based on Z-Score (Just for simulation/demo purposes)
        # If Z-score is high, evidence suggests it's overdue
        z = row['Z_Score']
        
        # Using a sigmoid-like modifier on the prior
        modifier = 1 / (1 + np.exp(-z)) 
        
        # Adjusted Posterior (Bayesian update simulation)
        posterior = prior * modifier * 2 
        # Cap at 50% for realism
        posterior = min(posterior, 0.5)
        
        results.append({
            'Numero': row['Numero'],
            'Prior': prior,
            'Posterior': posterior,
            'Z_Score_Evidencia': z
        })
        
    df_bayes = pd.DataFrame(results)
    return df_bayes
