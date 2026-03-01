import numpy as np
import pandas as pd
from scipy.special import comb

def calculate_system_payout(n_played, k_matches):
    """
    Calculates the detailed payout for a System Bet (Jugada Múltiple).
    Logic: If you play N numbers and hit K winning numbers, how many sub-combinations of 6
    have 3, 4, 5, or 6 hits?
    
    Formula: C(k_matches, x) * C(n_played - k_matches, 6 - x)
    where x is the prize tier (3, 4, 5, 6).
    """
    if k_matches < 3:
        return 0, {}
    
    # Prize Table (Per single combination)
    prizes = {
        3: 10,
        4: 100,
        5: 5000,
        6: 4000000 # Estimated Pozo
    }
    
    breakdown = {3:0, 4:0, 5:0, 6:0}
    total_winnings = 0
    
    for x in [3, 4, 5, 6]:
        # How many combinations of length 6 have exactly x hits?
        # We need to choose x from the k matched numbers AND (6-x) from the (n-k) non-matched numbers.
        if x <= k_matches and (6 - x) <= (n_played - k_matches):
            ways = comb(k_matches, x, exact=True) * comb(n_played - k_matches, 6 - x, exact=True)
            breakdown[x] = ways
            total_winnings += ways * prizes[x]
            
    return total_winnings, breakdown

def run_simulation(user_numbers, n_simulations=10000):
    """
    Run a vectorized Monte Carlo simulation for La Tinka.
    Args:
        user_numbers (list/set): The numbers chosen by the user (6 to 15).
        n_simulations (int): Number of simulated draws.
    """
    n_played = len(user_numbers)
    if n_played < 6 or n_played > 15:
        return None, 0, {}, 0

    user_set = np.array(list(user_numbers))
    
    # 1. Generate Winning Numbers Matrix (n_simulations x 6)
    rng = np.random.default_rng()
    sim_draws = np.zeros((n_simulations, 6), dtype=int)
    
    for i in range(n_simulations):
        sim_draws[i] = rng.choice(50, size=6, replace=False) + 1
        
    # 2. Vectorized Comparison (Count matches)
    # Check how many of user's N numbers satisfy the 6 drawn numbers
    matches = np.isin(sim_draws, user_set).sum(axis=1)
    
    # 3. Calculate Payouts per Simulation
    total_revenue = 0
    hit_counts = {3:0, 4:0, 5:0, 6:0}
    
    # Since we can have matches 0..6, we loop through unique match counts to optimize
    unique_matches, counts = np.unique(matches, return_counts=True)
    
    for m, count in zip(unique_matches, counts):
        if m >= 3:
            payout, breakdown = calculate_system_payout(n_played, m)
            total_revenue += payout * count
            
            # Aggregate "Jackpot" hits etc for simple display
            # Note: For system bets, a "5 match" outcome might actually contain many "3 match" prizes.
            # We track the highest tier reached per sim for the "Frequency" chart
            hit_counts[m] = hit_counts.get(m, 0) + count

    # 4. ROI Calculation
    # Cost Table
    cost_table = {
        6: 5, 7: 35, 8: 140, 9: 420, 10: 1050, 
        11: 2310, 12: 4620, 13: 8580, 14: 15015, 15: 25025
    }
    cost_per_play = cost_table.get(n_played, 5) # Fallback simplistic
    total_cost = n_simulations * cost_per_play
    
    roi_percent = ((total_revenue - total_cost) / total_cost) * 100
    
    return hit_counts, roi_percent, unique_matches, total_revenue

def get_kelly_criterion(win_prob, payout_ratio):
    """
    Calculates Kelly Criterion optimal bet size.
    f* = (bp - q) / b
    where:
    b = decimal odds - 1 (payout multiplier)
    p = probability of winning
    q = probability of losing (1 - p)
    """
    q = 1 - win_prob
    b = payout_ratio
    
    if b <= 0:
        return 0
        
    f_star = (b * win_prob - q) / b
    return max(0, f_star) # No shorting the lottery

def simulate_capital_growth(initial_capital, n_steps, win_prob, payout_ratio, strategy='kelly'):
    """
    Simulates capital growth over n steps using Kelly or Fixed betting.
    """
    capital = [initial_capital]
    f_star = get_kelly_criterion(win_prob, payout_ratio)
    
    for _ in range(n_steps):
        current_cap = capital[-1]
        
        if strategy == 'kelly':
            bet_size = current_cap * f_star
        elif strategy == 'fixed':
            bet_size = current_cap * 0.05 # 5% fixed
        else:
            bet_size = 5 # Fixed $5
            
        # Hard stop if ruin
        if current_cap <= 0:
            capital.append(0)
            continue
            
        win = np.random.random() < win_prob
        
        if win:
            new_cap = current_cap + (bet_size * payout_ratio)
        else:
            new_cap = current_cap - bet_size
            
        capital.append(new_cap)
        
    return capital

def run_ab_test_simulator(freq_df, n_future_draws=500):
    """
    Returns A/B test sequence comparing picking top 6 hot numbers vs random picks.
    """
    top_6 = freq_df.sort_values(by='Frecuencia', ascending=False).head(6)['Numero'].astype(int).values
    
    hot_hits = []
    random_hits = []
    
    # Vectorized future draws
    random_matrix = np.random.rand(n_future_draws, 50)
    future_draws = np.argsort(random_matrix, axis=1)[:, :6] + 1
    
    for draw in future_draws:
        # A strategy: Top 6 Hot Numbers
        hits_a = len(np.intersect1d(draw, top_6))
        hot_hits.append(hits_a)
        
        # B strategy: Fully Random Pick
        random_pick = np.random.choice(range(1, 51), 6, replace=False)
        hits_b = len(np.intersect1d(draw, random_pick))
        random_hits.append(hits_b)
        
    return pd.DataFrame({
        'Draw': range(1, n_future_draws + 1),
        'Hot_Strategy_Hits': hot_hits,
        'Random_Strategy_Hits': random_hits
    })
