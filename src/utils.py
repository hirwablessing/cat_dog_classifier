import os

def get_data_paths(base_path):
    train_dir = os.path.join(base_path, 'train')
    test_dir = os.path.join(base_path, 'test')
    return train_dir, test_dir
