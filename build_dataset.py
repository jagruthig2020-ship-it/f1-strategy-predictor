import pandas as pd
import os

print("🏁 Phase 1: Reading raw files...")
laps_path = os.path.join('data', 'lap_times.csv')
pits_path = os.path.join('data', 'pit_stops.csv')

laps_df = pd.read_csv(laps_path)
pits_df = pd.read_csv(pits_path)

laps_df.columns = laps_df.columns.str.strip()
pits_df.columns = pits_df.columns.str.strip()

print("🏁 Phase 1.5: Pre-processing Telemetry...")
laps_df['milliseconds'] = pd.to_numeric(laps_df['milliseconds'], errors='coerce')
laps_df['milliseconds'] = laps_df['milliseconds'].fillna(laps_df['milliseconds'].median())
laps_df['lap_time_secs'] = laps_df['milliseconds'] / 1000.0

print("🏁 Phase 2: Merging dataframes...")
merged_df = pd.merge(laps_df, pits_df, on=['raceId', 'driverId', 'lap'], how='left')
merged_df['pitted'] = merged_df['stop'].notna().astype(int)

print("🏁 Phase 3: Engineering live racing features...")
merged_df = merged_df.sort_values(by=['raceId', 'driverId', 'lap']).reset_index(drop=True)
merged_df['tyre_age'] = merged_df.groupby(['raceId', 'driverId']).cumcount() + 1

# Reset tire age when a pit stop happens
pit_mask = merged_df['pitted'] == 1
for idx in merged_df[pit_mask].index:
    race = merged_df.loc[idx, 'raceId']
    driver = merged_df.loc[idx, 'driverId']
    lap = merged_df.loc[idx, 'lap']
    reset_mask = (merged_df['raceId'] == race) & (merged_df['driverId'] == driver) & (merged_df['lap'] > lap)
    merged_df.loc[reset_mask, 'tyre_age'] = merged_df.loc[reset_mask, 'lap'] - lap

driver_race_avg = merged_df.groupby(['raceId', 'driverId'])['lap_time_secs'].transform('mean')
merged_df['lap_time_delta'] = merged_df['lap_time_secs'] - driver_race_avg

print("🏁 Phase 4: Engineering Track & Driver History...")
# Feature A: Historic average pit stop lap for this specific track (Track Profile)
track_pit_profile = merged_df[merged_df['pitted'] == 1].groupby('raceId')['lap'].mean().to_dict()
merged_df['track_avg_pit_lap'] = merged_df['raceId'].map(track_pit_profile).fillna(20)

# Feature B: Historic average tire life stint length for this driver (Driver Profile)
driver_pit_data = merged_df[merged_df['pitted'] == 1]
driver_stint_profile = driver_pit_data.groupby('driverId')['tyre_age'].mean().to_dict()
merged_df['driver_avg_stint_length'] = merged_df['driverId'].map(driver_stint_profile).fillna(15)

print("🏁 Phase 5: SHIFTING TARGET FOR ANTICIPATORY PREDICTION...")
# 🔥 CRITICAL FIX: Shift 'pitted' backward by 1 lap. 
# We are pairing Lap X's live telemetry with Lap X+1's pit status.
merged_df['predict_pit_next_lap'] = merged_df.groupby(['raceId', 'driverId'])['pitted'].shift(-1)

# Drop the last lap of the race for each driver since we don't know the "next" lap status
merged_df = merged_df.dropna(subset=['predict_pit_next_lap']).copy()
merged_df['predict_pit_next_lap'] = merged_df['predict_pit_next_lap'].astype(int)

# Save our predictive master dataset
output_path = os.path.join('data', 'f1_predictive_data.csv')
feature_cols = [
    'raceId', 'driverId', 'lap', 'position', 'tyre_age', 'lap_time_secs', 
    'lap_time_delta', 'track_avg_pit_lap', 'driver_avg_stint_length', 'predict_pit_next_lap'
]
merged_df[feature_cols].to_csv(output_path, index=False)
print(f"🎉 Success! Anticipatory predictive dataset saved to {output_path}")
