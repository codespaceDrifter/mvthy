import os
import re
import torch

def cleanup_checkpoints(folder, keep_last_n=5):
    def extract_batch(filename):
        match = re.search(r"batch_(\d+)", filename)
        if match:
            batch = int(match.group(1))
            return batch
        return -1
    
    files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    files_sorted = sorted(files, key=extract_batch, reverse=True)  # newest first
    
    for f in files_sorted[keep_last_n:]:
        os.remove(os.path.join(folder, f))


        
def check_bad_params(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"⚠️ Bad values detected in: {name}")
            return False
        else:
            print(f"{name}: Model Weights Healthy")
            return True

def sanitize_model_params(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        with torch.no_grad():
            bad_mask = torch.isnan(param) | torch.isinf(param)
            if bad_mask.any():
                print(f"Sanitizing {name}")
                param[bad_mask] = 0.0
                
            
def load_latest_checkpoint(folder, model):
    def extract_batch(filename):
        match = re.search(r"batch_(\d+)", filename)
        if match:
            batch = int(match.group(1))
            return batch
        return 0
    
    files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    if not files:
        return 0  # Start from beginning if no checkpoints
    
    files_sorted = sorted(files, key=extract_batch, reverse=True)  # newest first
    
    # Try loading latest checkpoint first
    try:
        latest_file = files_sorted[0]
        print(f"Loading latest checkpoint: {latest_file}")
        model.load_state_dict(torch.load(os.path.join(folder, latest_file)))

        sanitize_model_params(model)
        
        assert (check_bad_params(model))
        return extract_batch(latest_file)
    except Exception as e:
        print(f"Error loading latest checkpoint: {e}")
        if len(files_sorted) < 2:
            return 0  # Not enough checkpoints to try second latest
        
        second_latest_file = files_sorted[1]
        print(f"Loading second latest checkpoint: {second_latest_file}")
        model.load_state_dict(torch.load(os.path.join(folder, second_latest_file)))

        sanitize_model_params(model)
        
        assert (check_bad_params(model))
        return extract_batch(second_latest_file)

