import math

def calculate_entropy(data, target_idx):
    total = len(data)
    if total == 0: return 0.0
    counts = {}
    for row in data:
        val = row[target_idx]
        counts[val] = counts.get(val, 0) + 1
    
    entropy = 0.0
    for val in counts:
        p = counts[val] / total
        entropy -= p * math.log2(p)
    return entropy

def calculate_information_gain(data, feature_idx, target_idx):
    total_entropy = calculate_entropy(data, target_idx)
    total = len(data)
    subsets = {}
    for row in data:
        val = row[feature_idx]
        subsets.setdefault(val, []).append(row)
        
    weighted_entropy = 0.0
    for sub_data in subsets.values():
        weight = len(sub_data) / total
        weighted_entropy += weight * calculate_entropy(sub_data, target_idx)
        
    return total_entropy - weighted_entropy

def build_id3_tree(data, features_pool, target_idx, feature_names):
    targets = [row[target_idx] for row in data]
    if not targets: return 0
    majority_class = max(set(targets), key=targets.count)
    
    if len(set(targets)) == 1: return int(targets[0])
    if not features_pool: return int(majority_class)
        
    best_gain, best_feat = -1.0, None
    for feat in features_pool:
        gain = calculate_information_gain(data, feat, target_idx)
        if gain > best_gain:
            best_gain, best_feat = gain, feat
            
    if best_gain <= 0: return int(majority_class)
        
    branches = {}
    grouped = {}
    for row in data:
        grouped.setdefault(row[best_feat], []).append(row)
        
    remaining = [f for f in features_pool if f != best_feat]
    for val, sub_data in grouped.items():
        branches[str(val)] = build_id3_tree(sub_data, remaining, target_idx, feature_names)
        
    return {
        "feature": feature_names[best_feat],
        "feature_idx": best_feat,
        "branches": branches,
        "majority": int(majority_class),
        "type": "id3"
    }

def predict_id3(tree, sample_dict, path=None):
    if path is None: path = []
    if not isinstance(tree, dict): return int(tree), path
        
    feat_name = tree["feature"]
    val = str(sample_dict.get(feat_name))
    
    if val in tree["branches"]:
        path.append(f"{feat_name} == {val}")
        return predict_id3(tree["branches"][val], sample_dict, path)
        
    path.append(f"{feat_name} == {val} (unseen value fallback)")
    return int(tree["majority"]), path