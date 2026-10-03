import os
import urllib.request

os.makedirs("dataset/goemotions", exist_ok=True)

base_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"

files = [
    "train.tsv",
    "dev.tsv",
    "test.tsv",
    "emotions.txt"
]

for file_name in files:

    url = base_url + file_name

    save_path = os.path.join(
        "dataset",
        "goemotions",
        file_name
    )

    print("Downloading:", file_name)

    urllib.request.urlretrieve(
        url,
        save_path
    )

    print("Downloaded:", file_name)

print("\nGoEmotions dataset downloaded successfully!")