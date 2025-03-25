import numpy as np
import pandas as pd
import os

# Constants for thresholds
DISTANCE_THRESHOLD = 3.5
ANGLE_THRESHOLD_LOW = 135
ANGLE_THRESHOLD_HIGH = 180

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the CSV data, adjust the 'time' column, and create a combined 'node' column.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df.rename(columns={"Timeframe": "time"}, inplace=True)
    df['time'] = df['time'] - df['time'].min()
    df['node'] = df.apply(lambda row: f"{row['Atom_Name']}_{row['Residue_ID']}", axis=1)
    return df

def calculate_distance(row1, row2) -> float:
    """
    Calculate Euclidean distance between two rows based on X, Y, Z coordinates.
    """
    coord1 = np.array([row1.X, row1.Y, row1.Z])
    coord2 = np.array([row2.X, row2.Y, row2.Z])
    return np.linalg.norm(coord1 - coord2)

def calculate_angle(vec1, vec2) -> float:
    """
    Calculate the angle (in degrees) between two vectors.
    Returns NaN if one of the vectors has zero magnitude.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return np.nan  # Cannot define angle if a vector is zero-length
    cos_theta = np.dot(vec1, vec2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad)

def process_time_frame(t: float, df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame):
    """
    Process data for a single time frame t:
    - Compute distances between df1 and df2 rows.
    - Compute angles using df3 as reference.
    
    Returns:
        A tuple of lists (distance_results, angle_results).
    """
    distance_results = []
    angle_results = []
    
    df1_time = df1[df1['time'] == t].reset_index(drop=True)
    df2_time = df2[df2['time'] == t].reset_index(drop=False)  # preserve original index in 'Index'
    df3_time = df3[df3['time'] == t].reset_index(drop=False)
    
    if len(df2_time) != len(df3_time):
        print(f"Warning: For time {t}, df2 has {len(df2_time)} rows but df3 has {len(df3_time)} rows.")
    
    for row1 in df1_time.itertuples(index=False):
        for row2, row3 in zip(df2_time.itertuples(), df3_time.itertuples()):
            distance = calculate_distance(row1, row2)
            vec3_to_df1 = np.array([row1.X - row3.X, row1.Y - row3.Y, row1.Z - row3.Z])
            vec3_to_df2 = np.array([row2.X - row3.X, row2.Y - row3.Y, row2.Z - row3.Z])
            angle = calculate_angle(vec3_to_df1, vec3_to_df2)
            
            if distance <= DISTANCE_THRESHOLD:
                distance_results.append({
                    'source': row1.node,
                    'src_x': row1.X,
                    'src_y': row1.Y,
                    'src_z': row1.Z,
                    'src_mol': row1.Residue_ID,
                    'dst1': row2.node,
                    'dst1_x': row2.X,
                    'dst1_y': row2.Y,
                    'dst1_z': row2.Z,
                    'dst1_mol': row2.Residue_ID,
                    'dst': row2.Index,
                    'time': t,
                    'distance': distance
                })
            if ANGLE_THRESHOLD_LOW <= angle < ANGLE_THRESHOLD_HIGH:
                angle_results.append({
                    'source': row1.node,
                    'src_x': row1.X,
                    'src_y': row1.Y,
                    'src_z': row1.Z,
                    'src_mol': row1.Residue_ID,
                    'dst2': row3.node,
                    'dst2_x': row3.X,
                    'dst2_y': row3.Y,
                    'dst2_z': row3.Z,
                    'dst2_mol': row3.Residue_ID,
                    'dst': row3.Index,
                    'time': t,
                    'angle': angle
                })
    return distance_results, angle_results

def process_all_frames(df: pd.DataFrame):
    """
    Process all unique time frames and compile distance and angle results into DataFrames.
    """
    # Filter data based on conditions
    df1 = df[df['Atom_Type'].isin(['o', 'os'])].reset_index(drop=True)
    df2 = df[(df['Residue_Name'] == 'CSP') & (df['Atom_Type'] == 'n')].reset_index(drop=False)
    df3 = df[(df['Residue_Name'] == 'CSP') & (df['Atom_Type'] == 'hn')].reset_index(drop=False)
    
    all_distance_results = []
    all_angle_results = []
    
    for t in df1['time'].unique():
        d_res, a_res = process_time_frame(t, df1, df2, df3)
        all_distance_results.extend(d_res)
        all_angle_results.extend(a_res)
    
    return pd.DataFrame(all_distance_results), pd.DataFrame(all_angle_results)

def merge_results(df_distance: pd.DataFrame, df_angle: pd.DataFrame):
    """
    Merge distance and angle results on common keys, determine a label for each merged record,
    and adjust the destination node IDs so they don't overlap with source node IDs.
    """
    merged_df = pd.merge(df_distance, df_angle, 
                         on=['source', 'src_x', 'src_y', 'src_z', 'src_mol', 'time', 'dst'], 
                         how='outer')

    def determine_type(row):
        dst1_exists = pd.notnull(row.get('dst1'))
        dst2_exists = pd.notnull(row.get('dst2'))
        if dst1_exists and not dst2_exists:
            return 1
        elif not dst1_exists and dst2_exists:
            return 2
        elif dst1_exists and dst2_exists:
            return 3
        else:
            return np.nan

    merged_df['label'] = merged_df.apply(determine_type, axis=1)
    merged_df['src'] = pd.factorize(merged_df['source'])[0] + 1

    # Determine the offset based on the maximum source node ID
    max_src = merged_df['src'].max()
    offset = max_src + 1

    # Update the destination node IDs to start after the last source node
    merged_df['dst'] = merged_df['dst'] + offset

    merged_df = merged_df[['src','dst','src_mol', 'dst1_mol', 'dst2_mol', 'time', 'label']]
    return merged_df

def generate_node_features(merged_df: pd.DataFrame):
    """
    Generate and combine source and destination node features, then save them to a CSV.
    
    NOTE: This function assumes that merged_df already has destination node IDs adjusted 
    (i.e. the dst values include the offset from merge_results).
    """
    # Process source nodes
    src_features = merged_df[['src', 'src_mol']].drop_duplicates().rename(
        columns={'src': 'node', 'src_mol': 'feat'}
    )
    src_features['feat'] = src_features['feat'].astype(int)
    
    # Process destination nodes using the already offset dst values
    dst_features = merged_df[['dst', 'dst1_mol', 'dst2_mol']].drop_duplicates().rename(
        columns={'dst': 'node'}
    )
    dst_features['feat'] = dst_features.apply(
        lambda row: int(abs((row['dst1_mol'] if pd.notnull(row['dst1_mol']) else 0) -
                            (row['dst2_mol'] if pd.notnull(row['dst2_mol']) else 0))),
        axis=1
    )
    dst_features = dst_features[['node', 'feat']]
    
    node_features = pd.concat([src_features, dst_features], ignore_index=True)
    node_features = node_features.drop_duplicates(subset=['node']).reset_index(drop=True)
    return node_features

def process_and_save_dataframe(df, filename):
    """
    Process the input DataFrame to add 'ext_roll' and 'idx' columns,
    then save it to a CSV file.
    """
    df = df.copy()  # Ensure we're working on a copy to avoid SettingWithCopyWarning
    num_rows = len(df)

    # Initialize 'ext_roll' column with zeros using .loc for assignment clarity
    df.loc[:, 'ext_roll'] = 0

    # Assign 1 to the middle 15% rows and 2 to the last 15% rows
    df.loc[int(num_rows * 0.7):int(num_rows * 0.85) - 1, 'ext_roll'] = 1
    df.loc[int(num_rows * 0.85):, 'ext_roll'] = 2

    # Save the updated DataFrame
    df.to_csv(filename, index=False)
    print(f"Saved updated DataFrame to {filename}")

    return df

def main():
    filepath = "DATA/HB/HB100frames.csv"
    df = load_data(filepath)
    
    df_distance, df_angle = process_all_frames(df)
    merged_df = merge_results(df_distance, df_angle)
    
    # Add an index column and retain only required columns
    merged_df.insert(0, 'idx', range(len(merged_df)))
    final_merged_df = merged_df[['idx', 'src', 'dst', 'time', 'label']]
    
    node_features = generate_node_features(merged_df)
    node_features.to_csv('DATA/HB/node_features.csv', index=False)
    
    process_and_save_dataframe(final_merged_df, 'DATA/HB/edges.csv')

if __name__ == "__main__":
    main()