import math
from ml_core.id3 import calculate_entropy

def _split_info(partitions, total):
    si = 0.0
    for part in partitions:
        if not part: continue
        p = len(part) / total
        si -= p * math.log2(p)
    return si if si > 0 else 1e-9

def find_best_continuous_split(data, feature_idx, target_idx):
    total = len(data)
    total_entropy = calculate_entropy(data, target_idx)
    values = sorted(set(row[feature_idx] for row in data))
    
    if len(values) < 2: return 0.0, None, None
    best_gr, best_threshold, best_partitions = -1.0, None, None
    
    for i in range(len(values) - 1):
        threshold = (values[i] + values[i+1]) / 2.0
        left = [r for r in data if r[feature_idx] <= threshold]
        right = [r for r in data if r[feature_idx] > threshold]
        
        if not left or not right: continue
        
        weighted_entropy = (len(left)/total)*calculate_entropy(left, target_idx) + \
                           (len(right)/total)*calculate_entropy(right, target_idx)
        gain = total_entropy - weighted_entropy
        si = _split_info([left, right], total)
        gr = gain / si
        
        if gr > best_gr:
            best_gr, best_threshold, best_partitions = gr, threshold, {"<=": left, ">": right}
            
    return best_gr, best_threshold, best_partitions

def build_c45_tree(data, features_pool, target_idx, feature_names, continuous_features):
    targets = [row[target_idx] for row in data]
    if not targets: return 0
    majority_class = max(set(targets), key=targets.count)
    
    if len(set(targets)) == 1: return int(targets[0])
    if not features_pool: return int(majority_class)
    
    best_gr, best_feat, best_meta = -1.0, None, None
    
    for feat in features_pool:
        f_name = feature_names[feat]
        if f_name in continuous_features:
            gr, threshold, partitions = find_best_continuous_split(data, feat, target_idx)
            if gr > best_gr:
                best_gr, best_feat, best_meta = gr, feat, {"mode": "continuous", "threshold": threshold, "partitions": partitions}
        else:
            total_entropy = calculate_entropy(data, target_idx)
            subsets = {}
            for r in data: subsets.setdefault(r[feat], []).append(r)
            we = sum((len(s)/len(data))*calculate_entropy(s, target_idx) for s in subsets.values())
            gain = total_entropy - we
            si = _split_info(subsets.values(), len(data))
            gr = gain / si if si > 0 else 0.0
            if gr > best_gr:
                best_gr, best_feat, best_meta = gr, feat, {"mode": "categorical", "partitions": subsets}
                
    if best_gr <= 0 or best_feat is None: return int(majority_class)
        
    branches = {}
    f_name = feature_names[best_feat]
    
    if best_meta["mode"] == "continuous":
        remaining_features = features_pool
        for dynamic_key, sub_data in best_meta["partitions"].items():
            branches[dynamic_key] = build_c45_tree(sub_data, remaining_features, target_idx, feature_names, continuous_features)
        return {"feature": f_name, "mode": "continuous", "threshold": best_meta["threshold"], "branches": branches, "majority": int(majority_class), "type": "c45"}
    else:
        remaining_features = [f for f in features_pool if f != best_feat]
        for val, sub_data in best_meta["partitions"].items():
            branches[str(val)] = build_c45_tree(sub_data, remaining_features, target_idx, feature_names, continuous_features)
        return {"feature": f_name, "mode": "categorical", "branches": branches, "majority": int(majority_class), "type": "c45"}

def predict_c45(tree, sample_dict, path=None):
    if path is None: path = []
    if not isinstance(tree, dict): return int(tree), path
    
    f_name = tree["feature"]
    val = sample_dict.get(f_name)
    
    if tree["mode"] == "continuous":
        val_numeric = float(val)
        direction = "<=" if val_numeric <= tree["threshold"] else ">"
        path.append(f"{f_name} {direction} {round(tree['threshold'], 2)}")
        return predict_c45(tree["branches"][direction], sample_dict, path)
    else:
        val_str = str(val)
        if val_str in tree["branches"]:
            path.append(f"{f_name} == {val_str}")
            return predict_c45(tree["branches"][val_str], sample_dict, path)
        path.append(f"{f_name} == {val_str} (fallback)")
        return int(tree["majority"]), path