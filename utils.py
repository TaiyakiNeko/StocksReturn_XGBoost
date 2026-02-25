import os
import pandas as pd
import numpy as np
from numba import njit

# ==============================================================================
# Column Definitions
# ==============================================================================

col_names = [
    'Time', 'BidPrice1', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5', 
    'BidVolume1', 'BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5', 
    'AskPrice1', 'AskPrice2', 'AskPrice3', 'AskPrice4', 'AskPrice5', 
    'AskVolume1', 'AskVolume2', 'AskVolume3', 'AskVolume4', 'AskVolume5', 
    'OrderBuyNum', 'OrderSellNum', 'OrderBuyVolume', 'OrderSellVolume', 
    'TradeBuyNum', 'TradeSellNum', 'TradeBuyVolume', 'TradeSellVolume', 
    'TradeBuyAmount', 'TradeSellAmount', 'LastPrice'
]

col_indices = {name: i for i, name in enumerate(col_names)}

# Precomputed indices for JIT functions
bid_price_index = np.array([col_indices[f'BidPrice{i}'] for i in range(1, 6)])
ask_price_index = np.array([col_indices[f'AskPrice{i}'] for i in range(1, 6)])
bid_volume_index = np.array([col_indices[f'BidVolume{i}'] for i in range(1, 6)])
ask_volume_index = np.array([col_indices[f'AskVolume{i}'] for i in range(1, 6)])

# Single indices for JIT
idx_BidPrice1 = col_indices['BidPrice1']
idx_AskPrice1 = col_indices['AskPrice1']

# ==============================================================================
# Data Loading & Preprocessing
# ==============================================================================

def get_day_folders(data_path):
    """Retrieve sorted list of day folders."""
    folders = []
    for name in os.listdir(data_path):
        full_path = os.path.join(data_path, name)
        if os.path.isdir(full_path) and name.isdigit():
            folders.append(name)
    folders.sort(key=lambda x: int(x))
    return folders

def load_day_data(data_path, day_folder):
    """Load and clean CSV data for all stocks."""
    day_path = os.path.join(data_path, day_folder)
    data = {}
    
    for stock in ['A', 'B', 'C', 'D', 'E']:
        csv_path = os.path.join(day_path, f'{stock}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Identify Price columns
            price_cols = [c for c in df.columns if 'Price' in c]
            df[price_cols] = df[price_cols].astype(float)
            
            # Replace 0 with NaN in Price columns
            df[price_cols] = df[price_cols].replace(0.0, np.nan)
            
            # Forward fill and backward fill
            df[price_cols] = df[price_cols].ffill()
            df[price_cols] = df[price_cols].bfill()
            
            # Fix LastPrice if still NaN
            if 'LastPrice' in df.columns:
                mask = df['LastPrice'].isna()
                if mask.any():
                    df.loc[mask, 'LastPrice'] = df.loc[mask, 'BidPrice1']
                    df['LastPrice'] = df['LastPrice'].fillna(0.0)

            df = df.fillna(0.0)
            data[stock] = df.to_numpy()
        else:
            raise FileNotFoundError(f"Missing file: {csv_path}")
    return data

def clean_data(data):
    """Clean NaN and Inf values."""
    data = np.where(np.isnan(data), 0, data)
    data = np.where(np.isinf(data), 0, data)
    data = np.where(np.isinf(-data), 0, data)
    return data

def evaluate_ic(my_preds, ground_truth):
    """Calculate Information Coefficient."""
    data = np.vstack((my_preds, ground_truth))
    data = clean_data(data)
    cor = np.corrcoef(data)[0, 1]
    return cor

# ==============================================================================
# Microstructure Features (JIT Accelerated)
# ==============================================================================

@njit(fastmath=True, cache=True)
def compute_WAP_JIT(row):
    """
    Compute robust Weighted Average Price (WAP).
    Formula:
    $$ \\text{WAP} = \\frac{\\sum_{i=1}^{5} (P_{bid,i} \\cdot V_{ask,i}) + \\sum_{i=1}^{5} (P_{ask,i} \\cdot V_{bid,i})}{\\sum_{i=1}^{5} (V_{bid,i} + V_{ask,i})} $$
    """
    bid_value = np.sum(row[bid_price_index] * row[ask_volume_index])
    ask_value = np.sum(row[ask_price_index] * row[bid_volume_index])
    total_volume = np.sum(row[bid_volume_index] + row[ask_volume_index])
    return (bid_value + ask_value) / (total_volume + 1e-9)

@njit(fastmath=True, cache=True)
def compute_depth_imbalance_JIT(row):
    """
    Compute Order Book Depth Imbalance.
    Formula:
    $$ \\text{DI} = \\frac{\\sum V_{bid} - \\sum V_{ask}}{\\sum V_{bid} + \\sum V_{ask}} $$
    """
    total_bid = np.sum(row[bid_volume_index])
    total_ask = np.sum(row[ask_volume_index])
    return (total_bid - total_ask) / (total_bid + total_ask + 1e-9)

@njit(fastmath=True, cache=True)
def compute_relative_spread_JIT(row):
    """
    Compute Relative Spread.
    Formula:
    $$ \\text{Spread} = \\frac{P_{ask,1} - P_{bid,1}}{\\text{MidPrice}}, \\quad \\text{MidPrice} = \\frac{P_{ask,1} + P_{bid,1}}{2} $$
    """
    ask1 = row[idx_AskPrice1]
    bid1 = row[idx_BidPrice1]
    mid = (ask1 + bid1) / 2.0
    return (ask1 - bid1) / (mid + 1e-9)

@njit(fastmath=True, cache=True)
def compute_OFI_JIT(current_row, prev_row):
    """
    Compute Order Flow Imbalance (OFI).
    Formula:
    $$ \\text{OFI} = \\sum_{i=1}^{5} \\left( \\Delta V_{bid,i} \\cdot I(\\Delta P_{bid,i} \\geq 0) - \\Delta V_{ask,i} \\cdot I(\\Delta P_{ask,i} \\leq 0) \\right) $$
    """
    ofi = 0.0
    for i in range(5):
        # Bid Side
        delta_v_b = current_row[bid_volume_index[i]] - prev_row[bid_volume_index[i]]
        delta_p_b = current_row[bid_price_index[i]] - prev_row[bid_price_index[i]]
        if delta_p_b >= 0:
            ofi += delta_v_b
            
        # Ask Side
        delta_v_a = current_row[ask_volume_index[i]] - prev_row[ask_volume_index[i]]
        delta_p_a = current_row[ask_price_index[i]] - prev_row[ask_price_index[i]]
        if delta_p_a <= 0:
            ofi -= delta_v_a
            
    total_vol = np.sum(current_row[bid_volume_index] + current_row[ask_volume_index])
    return ofi / (total_vol + 1e-9)

@njit(fastmath=True, cache=True)
def compute_ob_slope_JIT(row):
    """
    Compute Order Book Slope.
    Formula:
    $$ \\text{Slope} = \\text{WeightedAvg}_{bid} - \\text{WeightedAvg}_{ask} $$
    $$ \\text{WeightedAvg} = \\frac{\\sum (w_i \\cdot V_i)}{\\sum V_i}, \\quad w_i = i $$
    """
    bid_slope_num = 0.0
    bid_slope_den = 0.0
    ask_slope_num = 0.0
    ask_slope_den = 0.0
    
    for i in range(5):
        weight = float(i + 1)
        bid_vol = row[bid_volume_index[i]]
        ask_vol = row[ask_volume_index[i]]
        
        bid_slope_num += weight * bid_vol
        bid_slope_den += bid_vol
        ask_slope_num += weight * ask_vol
        ask_slope_den += ask_vol
        
    bid_slope = bid_slope_num / (bid_slope_den + 1e-9)
    ask_slope = ask_slope_num / (ask_slope_den + 1e-9)
    
    return bid_slope - ask_slope

# ==============================================================================
# New Features for Sector Linkage (JIT Accelerated)
# ==============================================================================

@njit(fastmath=True, cache=True)
def compute_realized_volatility_JIT(returns_array):
    """
    Compute Realized Volatility (RV) over a window.
    Formula:
    $$ \\text{RV} = \\sqrt{\\sum_{t=1}^{T} r_t^2} $$
    """
    sum_sq = 0.0
    for r in returns_array:
        sum_sq += r * r
    return np.sqrt(sum_sq)

@njit(fastmath=True, cache=True)
def compute_rolling_corr_JIT(x_array, y_array):
    """
    Compute Pearson Correlation Coefficient over a window.
    Formula:
    $$ \\rho = \\frac{\\sum (x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum (x_i - \\bar{x})^2 \\sum (y_i - \\bar{y})^2}} $$
    """
    n = len(x_array)
    if n == 0:
        return 0.0
    
    mean_x = 0.0
    mean_y = 0.0
    for i in range(n):
        mean_x += x_array[i]
        mean_y += y_array[i]
    mean_x /= n
    mean_y /= n
    
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    
    for i in range(n):
        dx = x_array[i] - mean_x
        dy = y_array[i] - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
        
    den = np.sqrt(den_x * den_y)
    if den < 1e-9:
        return 0.0
    return num / den