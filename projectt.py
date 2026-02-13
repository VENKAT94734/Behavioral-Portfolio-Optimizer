import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
from sklearn.ensemble import RandomForestClassifier

# --- STEP 1: BIAS DETECTION ENGINE ---
# Features: [Avg_Trades_Per_Month, Avg_Hold_Days, Realized_Loss_Ratio]
# Labels: 0: Rational, 1: Overconfident, 2: Loss Averse
X_train = np.array([
    [2, 200, 0.5], [1, 350, 0.4],  # Rational
    [50, 5, 0.2], [40, 10, 0.1],   # Overconfident (High freq, low hold time)
    [3, 400, 0.05], [2, 500, 0.01] # Loss Averse (Low frequency, refuses to sell losers)
])
y_train = np.array([0, 0, 1, 1, 2, 2])

model = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)

def get_investor_bias(stats):
    pred = model.predict([stats])[0]
    biases = {0: "Rational", 1: "Overconfident", 2: "Loss Averse"}
    return biases[pred]

# --- STEP 2: PORTFOLIO OPTIMIZATION WITH BIAS PENALTY ---
def optimize_behavioral_portfolio(bias_type):
    # Mock Market Data (Tech, Gold, Bonds, S&P500)
    np.random.seed(42)
    assets = ["Tech", "Gold", "Bonds", "S&P500"]
    data = pd.DataFrame(np.random.normal(0.001, 0.02, (100, 4)), columns=assets)
    
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    ef = EfficientFrontier(mu, S)

    # ADJUST LOGIC BASED ON BIAS
    if bias_type == "Overconfident":
        # Add L2 Regularization to force diversification (prevents "betting the farm")
        ef.add_objective(objective_functions.L2_reg, gamma=2.0)
        nudge = "⚠️ Slow down! Your high trading frequency suggests overconfidence. We've diversified your picks to protect you from single-stock volatility."
        
    elif bias_type == "Loss Averse":
        # Add a constraint to limit max drawdown by favoring Bonds/Gold
        ef.add_constraint(lambda w: w[2] >= 0.30) # Minimum 30% in Bonds
        nudge = "🛡️ Safety first. Since you prefer avoiding losses, we've increased your Bond allocation to act as a buffer during market dips."
        
    else:
        ef.max_sharpe()
        nudge = "✅ You are trading rationally. Your portfolio is optimized for the best risk-adjusted returns."

    weights = ef.max_sharpe()
    return ef.clean_weights(), nudge

# --- STEP 3: EXECUTION (Simulating a New User) ---
# This user trades a lot (45/mo), holds for only 8 days, and rarely realizes losses.
current_user_stats = [45, 8, 0.15] 

user_bias = get_investor_bias(current_user_stats)
final_weights, personalized_nudge = optimize_behavioral_portfolio(user_bias)

print(f"--- PROJECT: BEHAVIORAL PORTFOLIO OPTIMIZER ---")
print(f"Detected Profile: {user_bias}")
print(f"\nRecommended Weights:")
for asset, w in final_weights.items():
    print(f"{asset}: {w*100:.1f}%")

print(f"\nINVESTOR NUDGE: {personalized_nudge}")
