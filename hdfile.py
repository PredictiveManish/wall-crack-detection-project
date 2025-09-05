import h5py

with h5py.File('crack_detector.h5', 'r') as f:
    print("Top-level keys:", list(f.keys()))
    if 'my_group' in f:
        group = f['my_group']
        print("Keys in 'my_group", list(group.keys()))
    if 'm'
