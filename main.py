import os
import pandas as pd
import numpy as np
import tqdm
from utils import *
from MyModel import MyModel

def run_test():
    # Load Model
    model = MyModel()
    
    # Load Day Data
    days = get_day_folders("./data")

    # Online Predict
    for day in days:
        model.reset()
        day_data = load_day_data("./data", day)
        
        E_array = day_data['E']
        n_ticks = len(E_array)
        ticktimes = E_array[:, 0]
        
        my_preds = np.zeros((n_ticks))

        print(f"Processing day {day}...")

        sector_arrays = [day_data[stock] for stock in ['A', 'B', 'C', 'D']]
        
        for tick_index in tqdm.tqdm(range(n_ticks)):
            E_row_data = E_array[tick_index]
            sector_row_datas = [
                sector_arrays[0][tick_index],
                sector_arrays[1][tick_index],
                sector_arrays[2][tick_index],
                sector_arrays[3][tick_index]
            ]
                
            # online_predict
            my_preds[tick_index] = model.online_predict(E_row_data, sector_row_datas)

        # Save Data
        output_dir = os.path.join(".", "output", day)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        out_frame = pd.DataFrame({'Time': ticktimes, 'Predict': my_preds})
        out_frame.to_csv(os.path.join(output_dir, "E.csv"), index=False)
        print (f"Saved output for Day {day}")

if __name__ == '__main__':
    run_test()

