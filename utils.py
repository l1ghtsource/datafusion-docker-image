import pandas as pd
from collections import defaultdict
import json


def analyze_category_tree(cat_df):
    children = defaultdict(list)
    for _, row in cat_df.iterrows():
        if pd.notna(row['parent_id']):
            children[row['parent_id']].append(row['cat_id'])

    def get_depth(cat_id, memo={}):
        if cat_id in memo:
            return memo[cat_id]

        parent = cat_df[cat_df['cat_id'] == cat_id]['parent_id'].values

        if len(parent) == 0 or pd.isna(parent[0]):
            memo[cat_id] = 0
            return 0
        
        depth = get_depth(parent[0], memo) + 1
        memo[cat_id] = depth
        return depth
    
    cat_df['depth'] = cat_df['cat_id'].apply(get_depth)
    
    def get_all_children(cat_id, all_children=None):
        if all_children is None:
            all_children = set()
        
        all_children.add(cat_id)
        for child in children.get(cat_id, []):
            get_all_children(child, all_children)
        
        return all_children
    
    def get_path_to_root(cat_id):
        path = [cat_id]
        current_id = cat_id
        
        while True:
            parent = cat_df[cat_df['cat_id'] == current_id]['parent_id'].values
            
            if len(parent) == 0 or pd.isna(parent[0]):
                break
                
            path.append(parent[0])
            current_id = parent[0]
        
        return path[::-1]
    
    cat_df['direct_children_count'] = cat_df['cat_id'].apply(lambda x: len(children.get(x, [])))
    
    cat_df['all_children_count'] = cat_df['cat_id'].apply(lambda x: len(get_all_children(x)) - 1)
    
    def get_full_path(cat_id):
        path_ids = get_path_to_root(cat_id)
        path_names = []
        
        for cid in path_ids:
            name = cat_df[cat_df['cat_id'] == cid]['cat_name'].values
            if len(name) > 0:
                path_names.append(name[0])
        
        return ' > '.join(path_names)
    
    cat_df['full_path'] = cat_df['cat_id'].apply(get_full_path)
    
    return cat_df


def parse_attr_to_dict(s):
    fixed_data_str = s.replace('""', '"')
    data = json.loads(fixed_data_str)
    
    result = {}
    for item in data:
        key = item.get('attribute_name')
        value = item.get('attribute_value', ''
        if 'attribute_measure' in item:
            value = f"{value} {item['attribute_measure']}"
        result[key] = value
    
    return result