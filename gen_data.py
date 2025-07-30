#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import torch
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# constants
DISTANCE_THRESHOLD = 3.5
ANGLE_THRESHOLD_LOW = 135
ANGLE_THRESHOLD_HIGH = 180

parser = argparse.ArgumentParser(
    description="Generate processed edge & node data from a raw CSV"
)
parser.add_argument(
    '--input',
    type=str,
    default="HB1000frames.csv",
    help="path to the raw CSV"
)

args = parser.parse_args()

# Load DataFrame with columns X, Y, Z, Atom_Name, Residue_Name, Residue_ID, Atom_Type, and Timeframe

if not os.path.exists(args.input):
    raise FileNotFoundError(f"File not found: {args.input_csv}")
df = pd.read_csv(args.input)
df.rename(columns={"Timeframe": "time"}, inplace=True)
df["time"] -= df["time"].min()
df["elem"] = df.apply(lambda r: f"{r['Atom_Name']}_{r['Residue_ID']}", axis=1)


# # Select all O Atoms
# df1 = df[df['Atom_Type'].isin(['o', 'os'])].reset_index(drop=True)
# # Pruning
df1 = df[(df['Residue_ID'].between(5, 14)) & df['Atom_Type'].isin(['o', 'os'])].reset_index(drop=True)
df2 = df[(df['Residue_Name'] == 'CSP') & (df['Atom_Type'] == 'n')].reset_index(drop=True)
df3 = df[(df['Residue_Name'] == 'CSP') & (df['Atom_Type'] == 'hn')].reset_index(drop=True)


def calculate_distance(row1, row2):
    """
    Calculate Euclidean distance between two rows based on X, Y, Z coordinates.
    
    Parameters:
        row1, row2: objects with attributes X, Y, Z.

    Returns:
        Euclidean distance (float).
    """
    coord1 = np.array([row1.X, row1.Y, row1.Z])
    coord2 = np.array([row2.X, row2.Y, row2.Z])
    return np.linalg.norm(coord1 - coord2)

def calculate_angle(vec1, vec2):
    """
    Calculate the angle between two vectors (in degrees).
    
    Parameters:
        vec1, vec2: numpy arrays representing the vectors.
    
    Returns:
        Angle between vec1 and vec2 in degrees (float).
    """
    # Compute cosine of the angle using the dot product
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    # Clip the value to avoid any numerical issues outside the valid range for arccos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad)


distance_results = []
angle_results = []

for t in df1['time'].unique():
    # Filter rows for the current time:
    df1_time = df1[df1['time'] == t].reset_index(drop=True)
    df2_time = df2[df2['time'] == t].reset_index()   
    df3_time = df3[df3['time'] == t].reset_index()   
    
    # For each row in df1, pair each row in df2 and df3 based on index
    for row1 in df1_time.itertuples(index=False):
        for row2, row3 in zip(df2_time.itertuples(), df3_time.itertuples()):
            # Calculate Euclidean distance between row1 (from df1) and row2 (from df2)
            distance = calculate_distance(row1, row2)
            
            # Calculate vectors from row3 (from df3) to row1 and row2
            vec3_to_df1 = np.array([row1.X - row3.X, row1.Y - row3.Y, row1.Z - row3.Z])
            vec3_to_df2 = np.array([row2.X - row3.X, row2.Y - row3.Y, row2.Z - row3.Z])
            angle = calculate_angle(vec3_to_df1, vec3_to_df2)
            
            # Append distance results if condition is met (distance <= 3.5)
            if distance <= 3.5:
                distance_results.append({
                    'source': row1.elem,
                    'src_x': row1.X,
                    'src_y': row1.Y,
                    'src_z': row1.Z,
                    'src_mol': row1.Residue_ID,
                    'dst': row2.elem,
                    'dst_x': row2.X,
                    'dst_y': row2.Y,
                    'dst_z': row2.Z,
                    'dst_mol': row2.Residue_ID,
                    'dst_idx': row2.Index,  
                    'time': t,
                    'distance': distance
                })
            # Append angle results if condition is met (135 <= angle < 180)
            if 135 <= angle < 180:
                angle_results.append({
                    'source': row1.elem,
                    'src_x': row1.X,
                    'src_y': row1.Y,
                    'src_z': row1.Z,
                    'src_mol': row1.Residue_ID,
                    'dst': row3.elem,
                    'dst_x': row3.X,
                    'dst_y': row3.Y,
                    'dst_z': row3.Z,
                    'dst_mol': row3.Residue_ID,
                    'dst_idx': row3.Index,  
                    'time': t,
                    'angle': angle
                })


df_distance = pd.DataFrame(distance_results)
df_angle = pd.DataFrame(angle_results)


keys = ["source","dst_idx","time"]


union = pd.concat([
    df_distance[keys].drop_duplicates(),
    df_angle   [keys].drop_duplicates()
], ignore_index=True)

counts = union.groupby(keys).size().reset_index(name="count")

counts["label"] = (counts["count"]==2).astype(int)


df_distance_common = df_distance.merge(counts[keys+["label"]], on=keys, how="left")
df_angle_common    = df_angle.merge(counts[keys+["label"]], on=keys, how="left")

df_distance_common["label"] = df_distance_common["label"].fillna(0).astype(int)
df_angle_common   ["label"] = df_angle_common   ["label"].fillna(0).astype(int)

df_distance_common = df_distance_common.reset_index(drop=True)
df_angle_common    = df_angle_common.reset_index(drop=True)


pos_df_distance = df_distance_common[df_distance_common["label"]==1]
neg_df_distance = df_distance_common[df_distance_common["label"]==0]

N = min(len(pos_df_distance), len(neg_df_distance))
pos_dist_samp = pos_df_distance.sample(n=N, random_state=0)
neg_dist_samp = neg_df_distance.sample(n=N, random_state=1)


df_distance_final = pd.concat([pos_dist_samp, neg_dist_samp]).sample(frac=1, random_state=2).reset_index(drop=True)

pos_df_angle = df_angle_common[df_angle_common["label"] == 1]
neg_df_angle = df_angle_common[df_angle_common["label"] == 0]


pos_ang_samp = pos_df_angle.sample(n=N, random_state=0)
neg_ang_samp = neg_df_angle.sample(n=N, random_state=1)

# Concatenate and shuffle
df_angle_final = (
    pd.concat([pos_ang_samp, neg_ang_samp])
      .sample(frac=1, random_state=2)
      .reset_index(drop=True)
)

# Quick sanity check
print("Distance final shape:", df_distance_final.shape)
print("Angle    final shape:", df_angle_final.shape)
print("Distance label counts:\n", df_distance_final["label"].value_counts())
print("Angle    label counts:\n", df_angle_final   ["label"].value_counts())

# Combine src and dst values for consistent mapping
combined_values = pd.concat([
    df_distance_final['source'], df_distance_final['dst'], 
    df_angle_final['source'], df_angle_final['dst']
])

# Use factorize to assign numeric indices starting from 0
numeric_indices, _ = pd.factorize(combined_values)

# Map src and dst directly to numeric indices using factorized output
mapping = pd.Series(numeric_indices, index=combined_values).to_dict()


for df in [df_distance_final, df_angle_final]:
    df['src'] = df['source'].map(mapping)
    df['dst'] = df['dst'].map(mapping)



def build_node_mapping(dfs, src_col='source', dst_col='dst'):
    """
    Given a list of DataFrames, returns a pd.Series mapping each unique node
    label in src_col or dst_col to a 0-based integer ID.
    """
    unique_nodes = pd.Index(
        pd.concat([df[src_col] for df in dfs] + [df[dst_col] for df in dfs])
          .unique()
    )
    return pd.Series(data=range(len(unique_nodes)), index=unique_nodes, name='node_id')

def remap_edges(df, mapping, src_col='source', dst_col='dst'):
    """
    Adds two new columns 'src' and 'dst' to df, mapping the original labels to ints.
    """
    df['src'] = df[src_col].map(mapping).astype(int)
    df['dst'] = df[dst_col].map(mapping).astype(int)

def generate_node_features(df1, df2, mapping, filename):
    """
    Extracts 'src_mol' and 'dst_mol' from both df1 & df2, merges them on the
    global node_id space (given by mapping), fills missing feats with 0,
    and writes a [1 x num_nodes] LongTensor to `filename`.
    """
    def collect(df, col, feat_col):
        tmp = (
            df[[col, feat_col]]
            .drop_duplicates()
            .rename(columns={col: 'node', feat_col: 'feat'})
        )
        tmp['node_id'] = tmp['node'].map(mapping).astype(int)
        return tmp[['node_id', 'feat']]

    f1 = collect(df1, 'source', 'src_mol')
    f2 = collect(df1, 'dst',    'dst_mol')
    f3 = collect(df2, 'source', 'src_mol')
    f4 = collect(df2, 'dst',    'dst_mol')

    feats = pd.concat([f1, f2, f3, f4], ignore_index=True)
    feats = feats.drop_duplicates(subset='node_id', keep='first')

    num_nodes = len(mapping)
    all_feat = pd.DataFrame({'node_id': range(num_nodes)})
    all_feat = all_feat.merge(feats, on='node_id', how='left')
    all_feat['feat'] = all_feat['feat'].fillna(0).astype(int)

    tensor = torch.tensor(all_feat['feat'].values, dtype=torch.long).unsqueeze(0)
    torch.save(tensor, filename)
    logging.info(f"Saved node features ({tensor.shape}) to {filename}")
    return tensor

def process_and_save_dataframe(df, filename):
    """
    Repackages df to columns [idx, src, dst, time, label, dst_idx, ext_roll],
    sorts by time, splits ext_roll, and writes to CSV.
    """
    df = df[['src','dst','time','label','dst_idx']].copy()
    df = df.sort_values('time', ascending=True, ignore_index=True)

    N = len(df)
    df.insert(0, 'idx', range(N))
    df['ext_roll'] = 0

    # assign 1 to the middle 15%, 2 to the last 15%
    df.loc[int(N*0.70):int(N*0.85), 'ext_roll'] = 1
    df.loc[int(N*0.85):,           'ext_roll'] = 2

    print(df['ext_roll'].value_counts(normalize=True))
    df.to_csv(filename, index=False)
    print(f"Saved updated DataFrame to {filename}")


mapping = build_node_mapping(
    [df_distance_final, df_angle_final],
    src_col='source', dst_col='dst',
)


node_feats = generate_node_features(
    df_distance_final,
    df_angle_final,
    mapping,
    filename='node_features.pt'
)


remap_edges(df_distance_final, mapping, src_col='source', dst_col='dst')
remap_edges(df_angle_final,    mapping, src_col='source', dst_col='dst')

process_and_save_dataframe(df_distance_final, 'edges1.csv')
process_and_save_dataframe(df_angle_final,    'edges2.csv')
