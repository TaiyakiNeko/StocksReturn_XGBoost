import os
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from utils import get_day_folders, load_day_data
from MyModel import MyModel


def build_training_data(horizon=5):
    data_path = "./data"
    if not os.path.exists(data_path):
        print(f"Data path {data_path} not found.")
        return None, None

    days = get_day_folders(data_path)
    model = MyModel()
    all_X = []
    all_y = []

    for day in days:
        day_data = load_day_data(data_path, day)
        X_day, y_day = process_day_data(day_data, model, horizon=horizon)
        if X_day.size == 0:
            continue
        all_X.append(X_day)
        all_y.append(y_day)

    if not all_X:
        return None, None

    return np.concatenate(all_X), np.concatenate(all_y)


def process_day_data(day_data, model, horizon=5):
    all_X = []
    all_y = []

    model.reset()
    target_df = day_data["E"]
    neighbors = ["A", "B", "C", "D"]

    if target_df.shape[1] < 31:
        return np.array([]), np.array([])

    n_ticks = len(target_df)
    if n_ticks <= horizon:
        return np.array([]), np.array([])

    for tick_index in range(n_ticks - horizon):
        target_row = target_df[tick_index]
        neighbor_rows = [day_data[s][tick_index] for s in neighbors]

        features = model.extract_features(target_row, neighbor_rows)
        features = np.clip(features, -10, 10)

        future_row = target_df[tick_index + horizon]
        cur_wap = model.compute_WAP(target_row)
        fut_wap = model.compute_WAP(future_row)
        y = np.log((fut_wap + 1e-9) / (cur_wap + 1e-9))

        all_X.append(features)
        all_y.append(y)

    X_day = np.array(all_X, dtype=np.float32)
    y_day = np.array(all_y, dtype=np.float32)
    print(f"Day data shape: X={X_day.shape}, y={y_day.shape}")
    return X_day, y_day


def build_dataset_from_days(days, horizon=5):
    data_path = "./data"
    if not os.path.exists(data_path):
        print(f"Data path {data_path} not found.")
        return None, None

    model = MyModel()
    all_X = []
    all_y = []

    for day in days:
        day_data = load_day_data(data_path, day)
        X_day, y_day = process_day_data(day_data, model, horizon=horizon)
        if X_day.size == 0:
            continue
        all_X.append(X_day)
        all_y.append(y_day)

    if not all_X:
        return None, None

    return np.concatenate(all_X), np.concatenate(all_y)

def analyze_feature_importance(model, top_n=20, feature_names=None):
    if not hasattr(model, 'feature_importances_'):
        print("Model does not support feature_importances_")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1] 

    print(f"\n{'='*20} Top {top_n} Feature Importances {'='*20}")
    print(f"{'Rank':<5} {'Feature Index':<15} {'Importance Score':<20}")
    print("-" * 50)
    
    for i in range(min(top_n, len(importances))):
        feat_idx = indices[i]
        feat_name = feature_names[feat_idx] if feature_names else f"Feature_{feat_idx}"
        score = importances[feat_idx]
        print(f"{i+1:<5} {feat_name:<15} {score:.6f}")
    print("=" * 50 + "\n")

    plt.figure(figsize=(10, 8))
    plt.title("Feature Importances (Gain)")
    
    if feature_names is None:
        labels = [f"Feature_{indices[i]}" for i in range(min(top_n, len(importances)))]
    else:
        labels = [feature_names[indices[i]] for i in range(min(top_n, len(importances)))]
    
    values = [importances[indices[i]] for i in range(min(top_n, len(importances)))]
    
    plt.barh(range(len(values)), values, align='center')
    plt.yticks(range(len(values)), labels)
    plt.xlabel("Average Gain")
    plt.gca().invert_yaxis() 

    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    plt.savefig("./plots/feature_importance.png")
    print("Feature importance plot saved to ./plots/feature_importance.png")
    plt.close()


def train():
    print("Building datasets...")
    data_path = "./data"
    days = get_day_folders(data_path)

    kfold_days = [d for d in days if d in ["1", "2", "3", "4"]]
    test_days = [d for d in days if d in ["5"]]

    # Load all Day1-4 data for KFold
    day_data_list = []
    for day in kfold_days:
        day_data = load_day_data("./data", day)
        X_day, y_day = process_day_data(day_data, MyModel(), horizon=5)
        if X_day.size == 0:
            continue
        day_data_list.append((X_day, y_day))

    # KFold: 4 folds, each time 1 day as val, 3 days as train
    print("Starting 4-Fold CV on Day1-4...")
    params = {
        "objective": "reg:absoluteerror",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 50,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
        "gamma": 2.0,
        "n_jobs": -1,
        "random_state": 42,
        "eval_metric": "mae",
        "early_stopping_rounds": 20,
    }

    fold_maes = []
    for fold in range(len(day_data_list)):
        # Prepare train/val split
        X_val, y_val = day_data_list[fold]
        X_train = np.concatenate([day_data_list[i][0] for i in range(len(day_data_list)) if i != fold])
        y_train = np.concatenate([day_data_list[i][1] for i in range(len(day_data_list)) if i != fold])

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            verbose=False,
            eval_set=[(X_val, y_val)],
        )
        preds = model.predict(X_val)
        mae = np.mean(np.abs(preds - y_val))
        fold_maes.append(mae)
        print(f"Fold {fold+1} (Day{kfold_days[fold]}) MAE: {mae:.6f}")

    print(f"Mean CV MAE: {np.mean(fold_maes):.6f}")

    # Train final model on Day1-3, validate on Day4, test on Day5
    train_days = [d for d in days if d in ["1", "2", "3"]]
    val_days = [d for d in days if d in ["4"]]
    X_train, y_train = build_dataset_from_days(train_days, horizon=5)
    X_val, y_val = build_dataset_from_days(val_days, horizon=5)
    X_test, y_test = build_dataset_from_days(test_days, horizon=5)

    if X_train is None or y_train is None:
        return

    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, "xgb_model.json")

    model = xgb.XGBRegressor(**params)
    eval_set = []
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]

    model.fit(
        X_train,
        y_train,
        verbose=False,
        eval_set=eval_set,
    )
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    if X_test is not None and y_test is not None:
        preds = model.predict(X_test)
        mae = np.mean(np.abs(preds - y_test))
        print(f"Test MAE: {mae:.6f}")

    analyze_feature_importance(model, top_n=20)


if __name__ == "__main__":
    train()