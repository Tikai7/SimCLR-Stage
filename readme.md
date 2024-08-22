# SimCLR Implementation for Historical Images

## Overview
This repository contains an implementation of SimCLR, a self-supervised learning model, designed to work with historical images. The primary objective is to retrieve the most similar image among a dataset by leveraging both visual and textual features.

## How It Works

### Step 1: Visual Similarity
1. Compute the feature vector for a given image.
2. Compute the feature vectors for all images in the database.
3. Calculate the cosine similarity between the feature vectors.

### Step 2: Textual Similarity
1. Generate the caption for the given image and encode it with BERT.
2. Generate captions for the images in the database and encode them with BERT.
3. Compute the cosine similarity between the textual embeddings.

### Step 3: Combine the Similarities
1. Perform a weighted combination of the visual and textual similarities.
2. Retrieve the top-K images with the highest combined similarity.

### Step 4: Refinement with SIFT Descriptor
Rearrange the top-K best images using the SIFT Descriptor for further refinement.

## Requirements

### Packages  : 
``pip install -r requirements.txt``

### Folders Structure
Please note that the provided dataloader is specifically built for these folders. If you use a different data structure, consider implementing a custom dataloader. The main contribution of this repository is the model implementation.

- `data/`
  - `data_pairs/`: Contains images with the target images extracted.
    - `pairs0..1919/`
  - `rol_sim_rol_pairs/`: Contains the target images corresponding to the images in the `sim_rol` folder.
    - `images.npy`
    - `targets.npy`
  - `rol/`: Contains images (e.g., `btv*.jpg`).
    - Example file name: `btv1b6904667b.jpg`
    - `json/`: Contains the JSON files with metadata.
    - `json_filtered/`: Contains JSON files with the relation field pointing to an image in the `sim_rol` folder.
    - `captions/`: Contains captions generated with BLIP.
    - `detailed-captions/`: Contains captions generated with Florence2.
  - `sim_rol_extracted/`: Contains images extracted from the `sim_rol` folder.
    - Example file name: `bpt6k6325514_f1_01_0.992.jpg`
    - Explanation:
      - `bpt*_f1`: Original name of the file in the `sim_rol` folder.
      - `01`: Indicates the first picture extracted from that journal.
      - `0.992`: Represents the extraction accuracy.
    - Subfolders:
      - `json/`: Contains JSON files with metadata.
      - `captions/`
      - `detailed-captions/`
  - `sim_rol_test/`: Contains test images.
    - Example file names: `bpt6k405973h_f564_02_0.919.jpg_ID_2.jpg` or `btv1b53224753t_ID_2`
    - Explanation:
      - `bpt6k405973h_f564_02_0.919`: Matches the file format in `sim_rol_extracted`.
      - `btv1b53224753t`: Matches the file format in `rol`.
      - `_ID_2`: Indicates that these two images are a pair because they share the same ID.
  - `files/`
    - `bad_pairs.txt`: Contains incorrectly matched images.
    - `to_enhance_pairs.txt`: Contains matches that can be enhanced by cropping the target image.
    - Example file name for both: `btv1b53218239v`
      - This is the name of the original image. To find the target image, refer to the `data_pairs` folder.

## How to Train the Model

1. Access the `TrainingSimCLR.py` file.
2. Update the file paths as needed:

    ```python
    path_rol_comp = "../data/rol" 
    path_sim_rol_extracted_comp = "../data/sim_rol_extracted" 
    path_filtered = "../data/rol/json_filtered"
    path_sim_rol_test = "../data/sim_rol_test"
    path_targets = "../data/rol_sim_rol_pairs/targets.npy"
    bad_pairs_path = "./data/files/bad_pairs.txt"
    to_enhance_path = "./data/files/to_enhance_pairs.txt"
    ```

3. Modify the training parameters if necessary:

    ```python
    epochs = 30
    image_size = 256
    batch_size = 64
    learning_rate = 1e-4
    train_ratio = 0.8
    val_ratio = 0.2
    temperature = 0.5
    ```

4. Load the data using the dataloader:

    ```python
    from utils.DataLoaderSimCLR import DataLoaderSimCLR as DSC
    
    dataset = DSC(
        path_rol_comp, path_sim_rol_extracted_comp, path_filtered, 
        shape=(image_size, image_size), target_path=path_targets,
        to_enhance_path=to_enhance_path, bad_pairs_path=bad_pairs_path,
        path_sim_rol_test=path_sim_rol_test, max_images=40000,
        augment_test=False, use_only_rol=True, remove_to_enhance_files=True, remove_bad_pairs=True
    )
    ```

## How to Test the Model

1. Access the `Matching.py` file.
2. Update the file paths as needed:

    ```python
    path_rol_comp = "../data/rol" 
    path_sim_rol_extracted_comp = "../data/sim_rol_extracted" 
    path_sim_rol_test = "../data/sim_rol_test"
    ```

3. Load the model:

    ```python
    model = SimCLR(feature_size=128)
    model_state = Trainer().get_model("path_to_model.pth")
    model.load_state_dict(model_state)
    ```

4. Load the test data:

    ```python
    testset = DLT(
        path_rol=path_rol_comp, path_sim_rol=path_sim_rol_extracted_comp,
        path_to_sim_test=path_sim_rol_test, augment=False, shape=(256,256)
    )
    ```

5. Find the top-K most similar images:

    ```python
    top_k_pairs, original_images, augmented_images, precisions = SMY.match_images_with_simCLR(
        model, test_loader=test_loader, k=10, use_sift=False, is_test=True, alpha=0.6
    )
    ```
