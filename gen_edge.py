#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist


# Helper Functions
def calculate_angle(vec1, vec2):
    """Calculate angle between two vectors."""
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle_rad)

def calculate_distance(row1, row2):
    """Calculate Euclidean distance."""
    return np.linalg.norm(np.array([row1.X, row1.Y, row1.Z]) - np.array([row2.X, row2.Y, row2.Z]))

def process_results(df):
    """Process DataFrame to add molecular IDs, unique nh_id, and labels."""
    df['src_mol'] = df['src'].apply(lambda x: int(x.split('_')[1]))
    df['dst_mol'] = df['dst'].apply(lambda x: int(x.split('_')[1]))
    df['nh_id'] = pd.factorize(df['dst'])[0] + 1
    df['label'] = np.where(
        (df['src_mol'].between(5, 14)) & (df['dst_mol'].between(1, 4)), 2, 1
    )
    return df

def process_and_save_dataframe(df, filename):
    """
    Save the processed DataFrame to a specified CSV file.

    Parameters:
    - df : Input DataFrame
    - filename: Output file path
    """
    num_rows = len(df)

    # Initialize 'ext_roll' column with zeros
    df['ext_roll'] = 0

    # Assign 1 to the middle 15% rows and 2 to the last 15% rows
    df.loc[int(num_rows * 0.7):int(num_rows * 0.85) - 1, 'ext_roll'] = 1
    df.loc[int(num_rows * 0.85):, 'ext_roll'] = 2

    # Insert an 'idx' column at the beginning
    df.insert(0, 'idx', range(len(df)))

    # Reindex and retain only the required columns
    df = df[['idx', 'src', 'dst', 'time', 'label', 'ext_roll', 'nh_id']]

    # Convert all columns to integers
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Save the updated DataFrame to the specified CSV file
    df.to_csv(filename, index=False)
    print(f"Saved updated DataFrame to {filename}")
    return df

def map_numeric_indices(dist_df, angle_df):
    """
    Map 'src' and 'dst' columns to numeric indices starting from 1 for consistent mapping.

    Parameters:
    - dist_df (pd.DataFrame): Distance DataFrame
    - angle_df (pd.DataFrame): Angle DataFrame

    Returns:
    - dist_df (pd.DataFrame): Updated Distance DataFrame with numeric indices
    - angle_df (pd.DataFrame): Updated Angle DataFrame with numeric indices
    """
    # Combine src and dst values for consistent mapping
    combined_values = pd.concat([
        dist_df['src'], dist_df['dst'], 
        angle_df['src'], angle_df['dst']
    ])

    # Use factorize to assign numeric indices starting from 0
    numeric_indices, _ = pd.factorize(combined_values)

    # Add 1 to ensure indices start from 1
    numeric_indices += 1

    # Map src and dst directly to numeric indices using factorized output
    mapping = pd.Series(numeric_indices, index=combined_values).to_dict()

    # Replace src and dst with numeric indices starting from 1
    for df in [dist_df, angle_df]:
        df['src'] = df['src'].map(mapping)
        df['dst'] = df['dst'].map(mapping)
    
    return dist_df, angle_df


####################################################

# DataFrame with X, Y, Z coordinates
filepath = "DATA/FILT_HB/HB100frames.csv"
df = pd.read_csv(filepath)
df.rename(columns={"Timeframe": "time"}, inplace=True)
df['time'] -= 255000
df['node'] = df.apply(lambda row: f"{row['Atom_Name']}_{row['Residue_ID']}", axis=1)

# Select relevant atoms and molecules
df1 = df[df['Atom_Name'].str.startswith('O')].reset_index(drop=True)
df2 = df[(df['Residue_Name'] == 'CSP') & (df['Atom_Type'] == 'n')].reset_index(drop=True)
df3 = df[(df['Residue_Name'] == 'CSP') & (df['Atom_Type'] == 'hn')].reset_index(drop=True)

# Calculate angles and distances
distance_results = []
angle_results = []

for t in df1['time'].unique():
    df1_time = df1[df1['time'] == t].reset_index(drop=True)
    df2_time = df2[df2['time'] == t].reset_index(drop=True)
    df3_time = df3[df3['time'] == t].reset_index(drop=True)

    for row1 in df1_time.itertuples(index=False):
        for idx, row3 in df3_time.iterrows():
            if idx < len(df2_time):
                row2 = df2_time.loc[idx]

                # Vectors and calculations
                vec3_to_df1 = np.array([row1.X - row3.X, row1.Y - row3.Y, row1.Z - row3.Z])
                vec3_to_df2 = np.array([row2['X'] - row3.X, row2['Y'] - row3.Y, row2['Z'] - row3.Z])
                angle = calculate_angle(vec3_to_df1, vec3_to_df2)
                distance = calculate_distance(row1, row2)

                # Append results
                distance_results.append({'src': row1.node, 'dst': row2['node'], 'time': t, 'distance': distance})
                angle_results.append({'src': row1.node, 'dst': row3['node'], 'time': t, 'angle': angle})

# Convert results to DataFrames
dist_df = process_results(pd.DataFrame(distance_results))
angle_df = process_results(pd.DataFrame(angle_results))


# Filter the distance DataFrame
filtered_dist_df = dist_df[dist_df['distance'] <= 3.5].copy()
filtered_dist_df.reset_index(drop=True, inplace=True)

# Filter the angle DataFrame
filtered_angle_df = angle_df[(angle_df['angle'] >= 135) & (angle_df['angle'] < 180)].copy()
filtered_angle_df.reset_index(drop=True, inplace=True)


# Find the intersection of rows based on 'time', 'src', and 'nh_id'
common_keys = pd.merge(
    filtered_dist_df[['time', 'src', 'nh_id']],
    filtered_angle_df[['time', 'src', 'nh_id']],
    on=['time', 'src', 'nh_id']
)

filtered_dist_df = filtered_dist_df.merge(common_keys, on=['time', 'src', 'nh_id'])
filtered_angle_df = filtered_angle_df.merge(common_keys, on=['time', 'src', 'nh_id'])


mapped_dist_df, mapped_angle_df = map_numeric_indices(filtered_dist_df, filtered_angle_df)


# Process and save distance DataFrame
final_dist_df = process_and_save_dataframe(mapped_dist_df, 'DATA/FILT_HB/edges1.csv')

# Process and save angle DataFrame
final_angle_df = process_and_save_dataframe(mapped_angle_df, 'DATA/FILT_HB/edges2.csv')

# Display the first few rows of both DataFrames
print("Updated Distance DataFrame:")
print(final_dist_df)

print("\nUpdated Angle DataFrame:")
print(final_angle_df)

