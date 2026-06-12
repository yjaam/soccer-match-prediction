import os
import pandas as pd
sd = os.path.dirname(os.path.abspath('m_Predict_New_Data/d_get_attendance_and_position.py'))
lp = os.path.join(sd, 'm_Predict_New_Data', 'data', 'lineups_values_ratings.csv')
print(f"Checking: {lp}")
print(f"Exists: {os.path.exists(lp)}")
