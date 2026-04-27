"""Download real-world datasets for SkillRulesEngine validation."""
import os
import subprocess
import sys
from pathlib import Path

def setup_kaggle():
    """Setup Kaggle CLI if not already configured."""
    kaggle_dir = Path.home() / ".kaggle"
    if not kaggle_dir.exists():
        print("Kaggle not configured. Please:")
        print("1. Go to kaggle.com/settings")
        print("2. Create API token")
        print("3. Place kaggle.json in ~/.kaggle/")
        return False
    return True

def download_resume_dataset():
    """Download Resume Dataset from Kaggle."""
    dataset_id = "gauravduttakiit/resume-dataset"
    output_dir = "data/raw/resume_dataset"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Download dataset
    cmd = ["kaggle", "datasets", "download", "-d", dataset_id, "-p", output_dir, "--unzip"]
    try:
        print(f"Downloading {dataset_id}...")
        subprocess.run(cmd, check=True)
        print(f"Downloaded {dataset_id} to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        return False
    except FileNotFoundError:
        print("kaggle CLI not found. Install with: pip install kaggle")
        return False

def download_bias_in_bios():
    """Download Bias-in-Bios dataset from Hugging Face."""
    try:
        from datasets import load_dataset
        output_dir = Path("data/raw/bias_in_bios")
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Downloading Bias-in-Bios dataset from Hugging Face...")
        dataset = load_dataset("LabHC/bias_in_bios")

        # Save to disk
        dataset.save_to_disk(str(output_dir))
        print(f"Downloaded Bias-in-Bios dataset to {output_dir}")
        return True
    except ImportError:
        print("datasets library not found. Install with: pip install datasets")
        return False
    except Exception as e:
        print(f"Failed to download Bias-in-Bios: {e}")
        return False

def verify_downloads():
    """Verify that datasets were downloaded successfully."""
    datasets = [
        ("Resume Dataset", "data/raw/resume_dataset"),
        ("Bias-in-Bios", "data/raw/bias_in_bios"),
    ]

    print("\n=== Download Verification ===")
    all_present = True
    for name, path in datasets:
        if Path(path).exists():
            file_count = len(list(Path(path).glob("**/*")))
            print(f"✓ {name}: {path} ({file_count} files)")
        else:
            print(f"✗ {name}: {path} (not found)")
            all_present = False

    return all_present

if __name__ == "__main__":
    print("=== Phase 1: Download Resume Datasets ===\n")

    # Try Kaggle first
    if setup_kaggle():
        download_resume_dataset()
    else:
        print("Skipping Kaggle dataset download (not configured)")

    # Try Hugging Face
    download_bias_in_bios()

    # Verify
    if verify_downloads():
        print("\nAll datasets downloaded successfully!")
        sys.exit(0)
    else:
        print("\nSome datasets missing. Please download manually or configure credentials.")
        sys.exit(1)
