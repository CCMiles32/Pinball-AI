import os
import random
import shutil

def split_dataset(base_dir='dataset', split_ratio=0.8, seed=42):
    random.seed(seed)

    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    random.shuffle(image_files)

    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    for split, files in [('train', train_files), ('val', val_files)]:
        img_dst = os.path.join(images_dir, split)
        lbl_dst = os.path.join(labels_dir, split)
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        for f in files:
            shutil.move(os.path.join(images_dir, f), os.path.join(img_dst, f))

            label_file = f.replace('.jpg', '.txt')
            label_path = os.path.join(labels_dir, label_file)
            if os.path.exists(label_path):
                shutil.move(label_path, os.path.join(lbl_dst, label_file))

    print(f"Dataset split complete: {len(train_files)} train / {len(val_files)} val")

split_dataset()
