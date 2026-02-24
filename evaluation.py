from utils import *

def run_evaluation():
    data_path = './data'
    output_path = os.path.join('./output')

    if not os.path.exists(output_path):
        print(f"Output path not found: {output_path}")
        return

    days = get_day_folders(output_path)
    print(f"{'交易日':<10} | {'IC 值':<10}")
    print("-" * 25)

    all_ic_values = []
    for day in days:
        pred_file = os.path.join(output_path, day, 'E.csv')
        pred_df = pd.read_csv(pred_file)

        truth_file = os.path.join(data_path, day, 'E.csv')
        truth_df = pd.read_csv(truth_file)

        ic = evaluate_ic(pred_df['Predict'], truth_df['Return5min'])

        all_ic_values.append(ic)
        print(f"{day:<10} | {ic:<10.4f}")
    
    mean_ic = np.mean(all_ic_values)
    print("-" * 25)
    print(f"{'Mean IC':<10} | {mean_ic:<10.4f}")

if __name__ == '__main__':
    # Evaluate IC
    run_evaluation()