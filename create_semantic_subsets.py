#!/usr/bin/env python3
"""Create semantic subsets of ImageNet-100 for diversity experiments.

Uses the known ImageNet-1K structure: classes 0-397 are animals,
classes 398-999 are non-animals (objects, food, structures, etc.).
"""
import os

MISC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "misc")

# Load all 1000 class IDs and names
with open(os.path.join(MISC_DIR, "class_indices.txt")) as f:
    all_ids = [l.strip() for l in f.readlines()]
with open(os.path.join(MISC_DIR, "class_names.txt")) as f:
    all_names = [l.strip() for l in f.readlines()]

# Load the 100-class subset
with open(os.path.join(MISC_DIR, "class100.txt")) as f:
    class100 = [l.strip() for l in f.readlines()]

# Map wordnet ID -> (index, name)
id_to_info = {}
for idx, (wid, name) in enumerate(zip(all_ids, all_names)):
    id_to_info[wid] = (idx, name)

# ImageNet-1K: indices 0-397 are animals, 398-999 are non-animals
ANIMAL_CUTOFF = 398

animals = []
non_animals = []
for wid in class100:
    idx, name = id_to_info.get(wid, (-1, "UNKNOWN"))
    if idx < ANIMAL_CUTOFF:
        animals.append((wid, idx, name))
    else:
        non_animals.append((wid, idx, name))

animals.sort(key=lambda x: x[1])
non_animals.sort(key=lambda x: x[1])

print(f"Total classes: {len(class100)}")
print(f"Animal classes (idx 0-397): {len(animals)}")
print(f"Non-animal classes (idx 398-999): {len(non_animals)}")
print("\n--- ANIMALS ---")
for wid, idx, name in animals:
    print(f"  [{idx:3d}] {wid}: {name}")
print("\n--- NON-ANIMALS ---")
for wid, idx, name in non_animals:
    print(f"  [{idx:3d}] {wid}: {name}")

# Write animal subset class file (keeping order from class100.txt)
animal_ids = [wid for wid, _, _ in animals]
with open(os.path.join(MISC_DIR, "class_animals_subset.txt"), "w") as f:
    for wid in animal_ids:
        f.write(wid + "\n")
print(f"\nWrote {len(animal_ids)} animal classes to class_animals_subset.txt")

# Write non-animal subset class file
non_animal_ids = [wid for wid, _, _ in non_animals]
with open(os.path.join(MISC_DIR, "class_objects_subset.txt"), "w") as f:
    for wid in non_animal_ids:
        f.write(wid + "\n")
print(f"Wrote {len(non_animal_ids)} non-animal classes to class_objects_subset.txt")
