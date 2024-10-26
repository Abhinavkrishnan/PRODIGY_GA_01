import os
from src.fine_tune import fine_tune_model

def main():
    print("Loading dataset...")
    dataset_path = os.path.join("data", "your_dataset.txt")
    # Load and process your dataset
    # ...

    print("Fine-tuning the model...")
    fine_tune_model(dataset_path)

if __name__ == "__main__":
    main()
