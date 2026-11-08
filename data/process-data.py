from pathlib import Path
import json

import pandas as pd
import numpy as np
import mediapipe as mp


# PATHS
project_root = Path.cwd().parent
repo_path = project_root / "data-repo"
metadata_path = repo_path / "metadata"
DATA_DIR = repo_path / "raw-data" / "20bn-jester-v1"
output_dir = repo_path / "target"

BATCH_SIZE = 32
MODEL_PATH = Path.cwd() / 'model/hand_landmarker.task'

TARGETS = [
    "Sliding Two Fingers Left",
    "Sliding Two Fingers Right",
    "Sliding Two Fingers Up",
    "Sliding Two Fingers Down",
    "No gesture",
    "Doing other things"
]

# LOAD DATA
train_folders = pd.read_csv(
    metadata_path / "train.csv",
    sep=";", header=None, names=["id", "label"]
)

validation_folders = pd.read_csv(
    metadata_path / "validation.csv",
    sep=";", header=None, names=["id", "label"]
)

test_folders = pd.read_csv(
    metadata_path / "test-answers.csv",
    sep=";", header=None, names=["id", "label"]
)

target_train = train_folders[train_folders["label"].isin(TARGETS)]
target_validation = validation_folders[
    validation_folders["label"].isin(TARGETS)
]
target_test = test_folders[test_folders["label"].isin(TARGETS)]


# FEATURE EXTRACTION
def mark(landmarker, img):
    mp_image = mp.Image.create_from_file(str(img))
    return landmarker.detect(mp_image)


def extract_features(marker_results):

    hand_landmarks = marker_results.hand_landmarks
    handedness = marker_results.handedness

    if not hand_landmarks:
        return np.append(np.zeros(63), -1.0)

    landmarks = np.array([
        coordinate
        for landmark in hand_landmarks[0]
        for coordinate in (landmark.x, landmark.y, landmark.z)
    ], dtype=np.float32)

    hand = handedness[0][0].category_name
    hand_value = 0.0 if hand == "Left" else 1.0

    return np.append(landmarks, hand_value)


# PREPROCESS
def preprocess(landmarker, batch, base_path):

    X_batch = []
    y_batch = []

    for folder in batch.itertuples(index=False):

        folder_path = base_path / str(folder.id)
        X_folder = []
        y_folder = str(folder.label)

        images = sorted(
            folder_path.glob("*.jpg"),
            key=lambda p: int(p.stem)
        )
        print("number of frames: ", len(images))

        for img in images:
            marked_img = mark(landmarker, img)
            features = extract_features(marked_img)
            X_folder.append(features)

        X_batch.append(X_folder)
        y_batch.append(y_folder)

    return (
        np.asarray(X_batch, dtype=object),
        np.asarray(y_batch)
    )


# BATCH PROCESSING
def process_batches(data, landmarker, data_dir, output_dir,
                    batch_size=32, start_batch=0):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_num in range(start_batch, (len(data) + batch_size - 1) // batch_size):

        start = batch_num * batch_size
        batch = data.iloc[start:start + batch_size]

        X, y = preprocess(landmarker, batch, data_dir)

        path = output_dir / f"batch_{batch_num}.npz"
        np.savez(path, X=X, y=y)

        yield batch_num, X, y


# CHECKPOINTING
def save_checkpoint(batch_number, output_dir):

    checkpoint_file = Path(output_dir) / "checkpoint.json"

    with open(checkpoint_file, "w") as f:
        json.dump({"last_completed_batch": batch_number}, f)


def load_checkpoint(output_dir):

    checkpoint_file = Path(output_dir) / "checkpoint.json"

    if not checkpoint_file.exists():
        return -1

    with open(checkpoint_file) as f:
        return json.load(f)["last_completed_batch"]


# PIPELINE
def process_pipeline(data, data_dir, output_dir, model_path, batch_size=32):

    last_completed = load_checkpoint(output_dir)
    start_batch = last_completed + 1

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
    )

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

        for batch_num, X, y in process_batches(
            data, landmarker, data_dir, output_dir,
            batch_size, start_batch
        ):

            print(f"Completed batch {batch_num + 1}")

            yield batch_num, X, y

def main():
  pipeline = process_pipeline(
    target_train, DATA_DIR, output_dir, MODEL_PATH
  )

  for batch_num, X, y in pipeline:

      # Process or inspect batch
      print(X.shape, y.shape)

      # Save checkpoint after successful processing
      save_checkpoint(batch_num, output_dir)

      if batch_num == 0:
          break
      

if __name__ == "__main__": 
    main()

            
