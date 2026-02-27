import numpy as np
import xgboost as xgb
from collections import deque
from utils import (
    col_indices, 
    compute_WAP_JIT, 
    compute_depth_imbalance_JIT, 
    compute_relative_spread_JIT, 
    compute_OFI_JIT, 
    compute_ob_slope_JIT,
    compute_realized_volatility_JIT,
    compute_rolling_corr_JIT
)

class MyModel:
    def __init__(self, model_path="./model/xgb_model.json", window_size=20):
        self.model = None
        self.model_path = model_path
        try:
            model = xgb.XGBRegressor()
            model.load_model(self.model_path)
            self.model = model
        except:
            print("Warning: Could not load model. This is fine if you are preparing data.")

        # State variables for WAP
        self.last_E_WAP = None
        self.last_sector_WAPs = [None, None, None, None]
        
        # State variables for Rows (to calculate true OFI)
        self.last_E_row = None
        self.last_sector_rows = [None, None, None, None]
        
        # State variables for OFI Momentum
        self.last_E_OFI = 0.0
        self.last_sector_OFIs = [0.0, 0.0, 0.0, 0.0]

        # Rolling buffers for Volatility and Correlation
        # Formula: Buffer stores $$ r_t $$ for past $$ T $$ steps
        self.window_size = window_size
        self.E_returns_buffer = deque(maxlen=window_size)
        self.Sector_returns_buffer = deque(maxlen=window_size)
        
        # Buffer for Lagged Sector Returns (Lead-Lag Effect)
        self.sector_return_lags = deque(maxlen=5)

    def reset(self):
        """Reset all state variables and buffers."""
        self.last_E_WAP = None
        self.last_sector_WAPs = [None, None, None, None]
        self.last_E_row = None
        self.last_sector_rows = [None, None, None, None]
        self.last_E_OFI = 0.0
        self.last_sector_OFIs = [0.0, 0.0, 0.0, 0.0]
        self.E_returns_buffer.clear()
        self.Sector_returns_buffer.clear()
        self.sector_return_lags.clear()

    def compute_WAP(self, row):
        return compute_WAP_JIT(row)

    def compute_depth_imbalance(self, row):
        return compute_depth_imbalance_JIT(row)

    def compute_relative_spread(self, row):
        return compute_relative_spread_JIT(row)

    def _update_buffers(self, E_log_return, mean_sector_log_return):
        """Update rolling buffers with new returns."""
        self.E_returns_buffer.append(E_log_return)
        self.Sector_returns_buffer.append(mean_sector_log_return)
        self.sector_return_lags.append(mean_sector_log_return)

    def _compute_advanced_features(self):
        """
        Compute advanced sector linkage features using JIT utilities.
        1. Volatility Spillover Ratio: $$ \\text{VR} = \\text{RV}_{own} / \\text{RV}_{sector} $$
        2. Rolling Correlation: $$ \\rho = \\text{Corr}(r_{own}, r_{sector}) $$
        3. Lagged Sector Return: $$ R_{sector, t-1} $$
        """
        if len(self.E_returns_buffer) < self.window_size:
            return 0.0, 0.0, 0.0
        
        # Convert deque to numpy array for JIT compatibility
        E_rvs = np.array(self.E_returns_buffer, dtype=np.float64)
        S_rvs = np.array(self.Sector_returns_buffer, dtype=np.float64)
        
        # 1. Realized Volatility
        rv_own = compute_realized_volatility_JIT(E_rvs)
        rv_sector = compute_realized_volatility_JIT(S_rvs)
        vol_ratio = rv_own / (rv_sector + 1e-9)
        
        # 2. Rolling Correlation
        corr_coef = compute_rolling_corr_JIT(E_rvs, S_rvs)
        if np.isnan(corr_coef):
            corr_coef = 0.0
            
        # 3. Lagged Sector Return (t-1)
        # The last element is current t, so -2 is t-1
        lag_1_sector_return = self.sector_return_lags[-2] if len(self.sector_return_lags) >= 2 else 0.0
        
        return vol_ratio, corr_coef, lag_1_sector_return

    def extract_features(self, E_row, sector_rows):
        # --- 1. WAP & Returns ---
        E_WAP = self.compute_WAP(E_row)
        sector_WAPs = [self.compute_WAP(row) for row in sector_rows]

        # Calculate Log Returns
        # Formula: $$ r_t = \\ln(P_t / P_{t-1}) $$
        E_log_return = (
            np.log(E_WAP / (self.last_E_WAP + 1e-9))
            if self.last_E_WAP
            else np.log(E_WAP / (E_row[col_indices['LastPrice']] + 1e-9))
        )
        sector_log_returns = [
            np.log(w / (lw + 1e-9)) if lw else np.log(w / (row[col_indices['LastPrice']] + 1e-9))
            for w, lw, row in zip(sector_WAPs, self.last_sector_WAPs, sector_rows)
        ]
        mean_sector_log_return = float(np.mean(sector_log_returns))

        # --- 2. Microstructure Features ---
        E_depth_imbalance = self.compute_depth_imbalance(E_row)
        sector_depth_imbalances = [self.compute_depth_imbalance(row) for row in sector_rows]
        mean_sector_depth_imbalance = float(np.mean(sector_depth_imbalances))

        E_order_imbalance = (
            (E_row[col_indices['OrderBuyVolume']] - E_row[col_indices['OrderSellVolume']])
            / (E_row[col_indices['OrderBuyVolume']] + E_row[col_indices['OrderSellVolume']] + 1e-9)
        )
        sector_order_imbalances = [
            (row[col_indices['OrderBuyVolume']] - row[col_indices['OrderSellVolume']]) /
            (row[col_indices['OrderBuyVolume']] + row[col_indices['OrderSellVolume']] + 1e-9)
            for row in sector_rows
        ]
        mean_sector_order_imbalance = float(np.mean(sector_order_imbalances))

        relative_spread = self.compute_relative_spread(E_row)

        # --- 3. OFI & Momentum ---
        # Calculate true OFI using stored previous rows
        if self.last_E_row is not None:
            E_OFI = compute_OFI_JIT(E_row, self.last_E_row)
            sector_OFIs = [
                compute_OFI_JIT(row, lrow) if lrow is not None else 0.0 
                for row, lrow in zip(sector_rows, self.last_sector_rows)
            ]
        else:
            # Fallback for first tick
            E_OFI = compute_OFI_JIT(E_row, E_row)
            sector_OFIs = [compute_OFI_JIT(row, row) for row in sector_rows]
            
        mean_sector_OFI = float(np.mean(sector_OFIs))
        
        # OFI Momentum (Delta OFI)
        # Formula: $$ \\Delta \\text{OFI} = \\text{OFI}_t - \\text{OFI}_{t-1} $$
        delta_E_OFI = E_OFI - self.last_E_OFI
        delta_sector_OFI = mean_sector_OFI - float(np.mean(self.last_sector_OFIs))

        # --- 4. Order Book Slope ---
        E_OB_slope = compute_ob_slope_JIT(E_row)
        sector_OB_slopes = [compute_ob_slope_JIT(row) for row in sector_rows]
        mean_sector_OB_slope = float(np.mean(sector_OB_slopes))

        # --- 5. Update Buffers & State ---
        self._update_buffers(E_log_return, mean_sector_log_return)
        
        # Update WAP State
        self.last_E_WAP = E_WAP
        self.last_sector_WAPs = sector_WAPs
        
        # Update Row State (for next OFI calc)
        self.last_E_row = E_row
        self.last_sector_rows = sector_rows
        
        # Update OFI State
        self.last_E_OFI = E_OFI
        self.last_sector_OFIs = sector_OFIs

        # --- 6. Compute Advanced Stats ---
        vol_ratio, corr_coef, lag_1_sector_return = self._compute_advanced_features()

        # --- 7. Construct Feature Vector ---
        return np.array(
            [
                # Basic Returns
                E_log_return,
                mean_sector_log_return,
                # Lead-Lag Feature
                lag_1_sector_return, 
                # Depth Imbalance
                E_depth_imbalance,
                mean_sector_depth_imbalance,
                # Order Imbalance
                E_order_imbalance,
                mean_sector_order_imbalance,
                # Spread
                relative_spread,
                # OFI
                E_OFI,
                mean_sector_OFI,
                # OFI Momentum
                delta_E_OFI,
                delta_sector_OFI,
                # OB Slope
                E_OB_slope,
                mean_sector_OB_slope,
                # Volatility Spillover
                vol_ratio,
                # Dynamic Correlation
                corr_coef,
                # Interaction Term (Sector Signal * Confidence)
                mean_sector_log_return * corr_coef 
            ],
            dtype=np.float32,
        )

    def online_predict(self, E_row, sector_rows):
        """Extract features and return prediction."""
        feature_vec = self.extract_features(E_row, sector_rows)
        return self.predict_from_features(feature_vec)

    def predict_from_features(self, feature_vec):
        """Run XGBoost prediction."""
        if self.model is None:
            return 0.0

        X_input = feature_vec.reshape(1, -1)
        X_input = np.clip(X_input, -10, 10)
        prediction = self.model.predict(X_input)
        return float(prediction[0] / 10000)