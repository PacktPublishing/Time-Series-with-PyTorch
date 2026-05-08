# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    """Dataset that accumulates (lookback, horizon) windows from each series with robust scaling."""
    def __init__(self, series_dict, lookback, horizon, scaler=None, is_train=True):
        self.lookback = lookback
        self.horizon = horizon
        self.samples = []
        self.is_train = is_train
        self.scaling_stats = {}
        
        # Create mapping from series ID to numerical index
        self.series_to_idx = {sid: i for i, sid in enumerate(series_dict.keys())}
        self.idx_to_series = {i: sid for sid, i in self.series_to_idx.items()}
        
        # Scale data if in training mode or use provided scaler
        if is_train:
            self.scaling_stats = self._compute_robust_scaling_stats(series_dict)
        elif scaler is not None:
            self.scaling_stats = scaler
            
        # Scale and process each series
        scaled_dict = self._scale_data(series_dict)
        
        # Process each scaled series
        for series_id, data in scaled_dict.items():
            if len(data) < lookback + horizon:
                continue
                
            for t in range(len(data) - lookback - horizon + 1):
                self.samples.append({
                    'series_idx': self.series_to_idx[series_id],
                    'input': data[t : t + lookback],
                    'target': data[t + lookback : t + lookback + horizon, 0],  # only 'sold'
                    'scaling_factors': np.array([
                        self.scaling_stats[series_id]['center'][0],  # center for 'sold'
                        self.scaling_stats[series_id]['scale'][0]    # scale for 'sold'
                    ])
                })

    def _compute_robust_scaling_stats(self, series_dict):
        """Compute robust scaling statistics for each series."""
        stats = {}
        for series_id, data in series_dict.items():
            # Use median and IQR for more robust scaling
            center = np.median(data, axis=0)
            q75, q25 = np.percentile(data, [75, 25], axis=0)
            iqr = q75 - q25
            # Use max(IQR, epsilon) to avoid division by zero
            scale = np.maximum(iqr, 1e-4)
            
            stats[series_id] = {
                'center': center,
                'scale': scale
            }
        return stats
    
    def _scale_data(self, series_dict):
        """Scale each series using robust statistics."""
        scaled_dict = {}
        for series_id, data in series_dict.items():
            if series_id in self.scaling_stats:
                center = self.scaling_stats[series_id]['center']
                scale = self.scaling_stats[series_id]['scale']
                scaled_dict[series_id] = (data - center) / scale
                # Clip extreme values to prevent instability
                scaled_dict[series_id] = np.clip(scaled_dict[series_id], -10, 10)
            else:
                scaled_dict[series_id] = data
        return scaled_dict
    
    def get_scaling_stats(self):
        """Return scaling statistics for use in validation/test sets."""
        return self.scaling_stats
    
    def inverse_transform(self, data, series_idx):
        """Transform data back to original scale."""
        series_id = self.idx_to_series[series_idx]
        if series_id in self.scaling_stats:
            center = self.scaling_stats[series_id]['center'][0]
            scale = self.scaling_stats[series_id]['scale'][0]
            return data * scale + center
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'series_idx': torch.LongTensor([sample['series_idx']]),
            'input': torch.FloatTensor(sample['input']),
            'target': torch.FloatTensor(sample['target']),
            'scaling_factors': torch.FloatTensor(sample['scaling_factors'])
        }