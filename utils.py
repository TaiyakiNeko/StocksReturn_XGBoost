import os
import pandas as pd
import numpy as np
from numba import njit

# Column definitions for numpy
col_names = ['Time', 'BidPrice1', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5', 
             'BidVolume1', 'BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5', 
             'AskPrice1', 'AskPrice2', 'AskPrice3', 'AskPrice4', 'AskPrice5', 
             'AskVolume1', 'AskVolume2', 'AskVolume3', 'AskVolume4', 'AskVolume5', 
             'OrderBuyNum', 'OrderSellNum', 'OrderBuyVolume', 'OrderSellVolume', 
             'TradeBuyNum', 'TradeSellNum', 'TradeBuyVolume', 'TradeSellVolume', 
             'TradeBuyAmount', 'TradeSellAmount', 'LastPrice']

col_indices = {name: i for i, name in enumerate(col_names)}

# Precomputed indices for JIT functions
bid_price_index = np.array([col_indices[f'BidPrice{i}'] for i in range(1, 6)])
ask_price_index = np.array([col_indices[f'AskPrice{i}'] for i in range(1, 6)])
bid_volume_index = np.array([col_indices[f'BidVolume{i}'] for i in range(1, 6)])
ask_volume_index = np.array([col_indices[f'AskVolume{i}'] for i in range(1, 6)])

# Single indices for JIT
idx_BidPrice1 = col_indices['BidPrice1']
idx_AskPrice1 = col_indices['AskPrice1']

def get_day_folders(data_path):
    folders = []
    for name in os.listdir(data_path):
        full_path = os.path.join(data_path, name)
        if os.path.isdir(full_path) and name.isdigit():
            folders.append(name)
    folders.sort(key=lambda x: int(x))
    return folders

def load_day_data(data_path, day_folder):
    day_path = os.path.join(data_path, day_folder)
    data = {}
    
    # Pre-computation of dataframe ops
    for stock in ['A', 'B', 'C', 'D', 'E']:
        csv_path = os.path.join(day_path, f'{stock}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Identify Price columns (Bid/Ask/Last) which should never be 0
            # Volume can be 0, Price cannot.
            # Convert to float first to allow np.nan
            price_cols = [c for c in df.columns if 'Price' in c]
            df[price_cols] = df[price_cols].astype(float)
            
            # Replace 0 with NaN only in Price columns
            # Also replace 0.0 only. If it's already NaN, it stays NaN
            df[price_cols] = df[price_cols].replace(0.0, np.nan)
            
            # Forward fill to propagate last valid price
            df[price_cols] = df[price_cols].ffill()
            df[price_cols] = df[price_cols].bfill() # Handle initial 0s
            
            # CRITICAL FIX for Day 1/5: 
            # If LastPrice was 0 and ffill didn't work (e.g. start of day), fill with BidPrice1
            if 'LastPrice' in df.columns:
                 mask = df['LastPrice'].isna()
                 if mask.any():
                     df.loc[mask, 'LastPrice'] = df.loc[mask, 'BidPrice1']
                     # If BidPrice1 was also NaN (unlikely but possible), fill with 0 (bad but no choice)
                     df['LastPrice'] = df['LastPrice'].fillna(0.0)

            # Fill any remaining NaNs (e.g. if column was all 0) with 0 or mean
            # Ideally shouldn't happen for valid stocks
            df = df.fillna(0.0)
            
            data[stock] = df.to_numpy()
        else:
            raise FileNotFoundError(f"Missing file: {csv_path}")
    return data


def clean_data(data):
    data = np.where(np.isnan(data), 0, data)
    data = np.where(np.isinf(data), 0, data)
    data = np.where(np.isinf(-data), 0, data)
    return data


def evaluate_ic(my_preds, ground_truth):
    data = np.vstack((my_preds, ground_truth))
    data = clean_data(data)
    cor = np.corrcoef(data)[0, 1]
    return cor
      
@njit(fastmath=True, cache=True)
def compute_WAP_JIT(row):
    """
    Compute robust WAP using all 5 levels of order book.
    Formula:
        numerator   = Σ(BidPrice_i * AskVolume_i) + Σ(AskPrice_i * BidVolume_i)
        denominator = Σ(BidVolume_i + AskVolume_i)
        This serves as the primary fair price proxy.
    """
    bid_value = np.sum(row[bid_price_index] * row[ask_volume_index])
    ask_value = np.sum(row[ask_price_index] * row[bid_volume_index])
    total_volume = np.sum(row[bid_volume_index] + row[ask_volume_index])
    return (bid_value + ask_value) / (total_volume + 1e-9)

@njit(fastmath=True, cache=True)
def compute_depth_imbalance_JIT(row):
    """
    Compute depth imbalance as:
        Depth Imbalance = (Σ(BidVolume_i) - Σ(AskVolume_i)) / (Σ(BidVolume_i) + Σ(AskVolume_i))
    This captures the overall market pressure.
    """
    total_bid = np.sum(row[bid_volume_index])
    total_ask = np.sum(row[ask_volume_index])
    return (total_bid - total_ask) / (total_bid + total_ask + 1e-9)

@njit(fastmath=True, cache=True)
def compute_relative_spread_JIT(row):
    """
    Compute relative spread as:
        Relative Spread = (AskPrice1 - BidPrice1) / MidPrice
    MidPrice = (AskPrice1 + BidPrice1) / 2
    """
    ask1 = row[idx_AskPrice1]
    bid1 = row[idx_BidPrice1]
    mid = (ask1 + bid1) / 2.0
    return (ask1 - bid1) / (mid + 1e-9)

@njit(fastmath=True, cache=True)
def compute_OFI_JIT(current_row, prev_row):
    """
    Compute Order Flow Imbalance
    Formula: OFI = Σ(ΔBidVol * I(ΔBidPrice >= 0)) - Σ(ΔAskVol * I(ΔAskPrice <= 0))
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
    Compute Order Book Slope
    Formula: Slope = WeightedAvg(BidLevel) - WeightedAvg(AskLevel)
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