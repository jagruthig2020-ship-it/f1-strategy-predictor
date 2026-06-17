import pandas as pd
import os

print("🏁 Phase 1: Reading raw files...")
laps_path = os.path.join('data', 'lap_times.csv')
pits_path = os.path.join('data', 'pit_stops.csv')

laps_df = pd.read_csv(laps_path)
pits_df = pd.read_csv(pits_path)

# 🔥 FIX 1: Strip any accidental hidden spaces from the CSV column headers
laps_df.columns = laps_df.columns.str.strip()
pits_df.columns = pits_df.columns.str.strip()

print("🏁 Phase 1.5: Pre-processing Telemetry...")
# 🔥 FIX 2: Convert milliseconds to numeric BEFORE we do any merging
laps_df['milliseconds'] = pd.to_numeric(laps_df['milliseconds'], errors='coerce')
laps_df['milliseconds'] = laps_df['milliseconds'].fillna(laps_df['milliseconds'].median())
laps_df['lap_time_secs'] = laps_df['milliseconds'] / 1000.0

print("🏁 Phase 2: Merging dataframes...")
# Now we merge safely. laps_df already has our clean 'lap_time_secs'
merged_df = pd.merge(laps_df, pits_df, on=['raceId', 'driverId', 'lap'], how='left')
merged_df['pitted'] = merged_df['stop'].notna().astype(int)

print("🏁 Phase 3: Engineering 'tyre_age' feature...")
merged_df = merged_df.sort_values(by=['raceId', 'driverId', 'lap']).reset_index(drop=True)
merged_df['tyre_age'] = merged_df.groupby(['raceId', 'driverId']).cumcount() + 1

pit_mask = merged_df['pitted'] == 1
for idx in merged_df[pit_mask].index:
    race = merged_df.loc[idx, 'raceId']
    driver = merged_df.loc[idx, 'driverId']
    lap = merged_df.loc[idx, 'lap']
    reset_mask = (merged_df['raceId'] == race) & (merged_df['driverId'] == driver) & (merged_df['lap'] > lap)
    merged_df.loc[reset_mask, 'tyre_age'] = merged_df.loc[reset_mask, 'lap'] - lap

print("🏁 Phase 3.5: Engineering Delta Features...")
# Calculate the personal average lap time for that driver in THAT specific race
driver_race_avg = merged_df.groupby(['raceId', 'driverId'])['lap_time_secs'].transform('mean')

# Delta Time: How much slower/faster is this lap compared to their average pace?
merged_df['lap_time_delta'] = merged_df['lap_time_secs'] - driver_race_avg

# Save our newly enriched engineered master dataset
output_path = os.path.join('data', 'f1_processed_data.csv')
feature_cols = ['raceId', 'driverId', 'lap', 'position', 'tyre_age', 'lap_time_secs', 'lap_time_delta', 'pitted']
merged_df[feature_cols].to_csv(output_path, index=False)

print(f"🎉 Success! Enriched master dataset saved to {output_path}")