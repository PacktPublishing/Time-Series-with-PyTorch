#main.py

from dataset import TimeSeriesDataset
import torch
from torch.utils.data import DataLoader 

def create_optimized_dataloaders(
    train_dict, 
    test_dict, 
    lookback, 
    horizon, 
    batch_size, 
    val_split=0.2,
    num_workers=4
):
    """Creates DataLoaders with optimized parallel processing configuration and scaling."""
    # Create training dataset first to get scaling statistics
    train_dataset = TimeSeriesDataset(train_dict, lookback, horizon, is_train=True)
    scaling_stats = train_dataset.get_scaling_stats()
    
    # Create test dataset using training scaling statistics
    test_dataset = TimeSeriesDataset(
        test_dict, 
        lookback, 
        horizon, 
        scaler=scaling_stats,
        is_train=False
    )
    
    # Split test into validation and test
    test_size = len(test_dataset)
    val_size = int(test_size * val_split)
    test_size = test_size - val_size
    
    val_dataset, final_test_dataset = torch.utils.data.random_split(
        test_dataset, 
        [val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Optimized DataLoader configuration
    loader_config = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 2,
        'shuffle': False
    }
    
    # Create loaders with optimized settings
    train_loader = DataLoader(train_dataset, **loader_config)
    val_loader = DataLoader(val_dataset, **loader_config)
    test_loader = DataLoader(final_test_dataset, **loader_config)
    
    print(f"\nDataLoader Configuration:")
    print(f"Number of workers: {num_workers}")
    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

# Usage example
"""if __name__ == "__main__":
    # Load your data first
    train_dict, test_dict = load_m5_data(m5_csv)
    
    # Create dataloaders with 4 workers
    train_loader, val_loader, test_loader = create_optimized_dataloaders(
        train_dict=train_dict,
        test_dict=test_dict,
        lookback=14,
        horizon=1,
        batch_size=32,
        num_workers=4
    )"""