
import os
import gdown

CHECKPOINT_URL = "https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH"
CHECKPOINT_DIR = "external/SCHP/checkpoints"
CHECKPOINT_NAME = "exp-schp-201908261155-lip.pth"


def download_lip_checkpoint():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

    if os.path.exists(ckpt_path):
        print(f"Checkpoint already exists at {ckpt_path}, skipping download.")
        return ckpt_path

    print("Downloading SCHP LIP checkpoint from Google Drive...")
    gdown.download(CHECKPOINT_URL, ckpt_path, quiet=False)
    return ckpt_path


if __name__ == "__main__":
    download_lip_checkpoint()
