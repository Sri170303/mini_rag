# utils.py
import requests
import os
from config import DATA_DIR

def fetch_google_drive_md(url):
    """Fetch a .md file from Google Drive public link."""
    if "drive.google.com" in url:
        file_id = url.split("/file/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(download_url)
        if response.status_code == 200:
            os.makedirs(DATA_DIR, exist_ok=True)
            filename = f"drive_file_{file_id}.md"
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(response.text)
            return filepath
        else:
            raise Exception(f"Failed to fetch document: HTTP {response.status_code}")
    else:
        raise Exception("Invalid Google Drive URL")