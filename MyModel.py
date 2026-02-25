import numpy as np
import xgboost as xgb
from utils import col_indices, compute_WAP_JIT, compute_depth_imbalance_JIT, compute_relative_spread_JIT


class MyModel:
    def __init__(self, model_path="./model/xgb_model.json"):
        self.model = None
        self.model_path = model_path
        try:
            model = xgb.XGBRegressor()
            model.load_model(self.model_path)
            self.model = model
        except:
            print("Warning: Could not load model. This is fine if you are preparing data.")

        self.last_E_WAP = None
        self.last_sector_WAPs = [None, None, None, None]

    def reset(self):
        self.last_E_WAP = None
        self.last_sector_WAPs = [None, None, None, None]

    def compute_WAP(self, row):
        return compute_WAP_JIT(row)

    def compute_depth_imbalance(self, row):
        return compute_depth_imbalance_JIT(row)

    def compute_relative_spread(self, row):
        return compute_relative_spread_JIT(row)

    def extract_features(self, E_row, sector_rows):
        E_WAP = self.compute_WAP(E_row)
        sector_WAPs = [self.compute_WAP(row) for row in sector_rows]

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

        self.last_E_WAP = E_WAP
        self.last_sector_WAPs = sector_WAPs

        return np.array(
            [
                E_log_return,
                mean_sector_log_return,
                E_depth_imbalance,
                mean_sector_depth_imbalance,
                E_order_imbalance,
                mean_sector_order_imbalance,
                relative_spread,
            ],
            dtype=np.float32,
        )

    def online_predict(self, E_row, sector_rows):
        feature_vec = self.extract_features(E_row, sector_rows)
        return self.predict_from_features(feature_vec)

    def predict_from_features(self, feature_vec):
        if self.model is None:
            return 0.0

        X_input = feature_vec.reshape(1, -1)
        X_input = np.clip(X_input, -10, 10)
        prediction = self.model.predict(X_input)
        return float(prediction[0])
