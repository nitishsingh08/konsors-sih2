
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE="$ROOT_DIR/Features_mapped.7z"
DEST="$ROOT_DIR/Features_mapped"
FILE_ID="YOUR_GOOGLE_DRIVE_FILE_ID"

if [[ -d "$DEST" ]]; then
    echo "Features_mapped already exists; skipping extraction."
    exit 0
fi

if [[ ! -f "$ARCHIVE" ]]; then
    echo "Downloading Features_mapped.7z from Google Drive..."

    URL="https://drive.google.com/drive/folders/1OFclk-M-7VIJJjl1ibMvZ6cIYVnv1O4n?usp=sharing"

    if command -v curl >/dev/null 2>&1; then
        curl -fL "$URL" -o "$ARCHIVE"
    else
        echo "Please install curl to download the archive." >&2
        exit 1
    fi
fi

if command -v 7zz >/dev/null 2>&1; then
    SEVENZIP=7zz
elif command -v 7z >/dev/null 2>&1; then
    SEVENZIP=7z
else
    echo "Install 7-Zip first: brew install sevenzip" >&2
    exit 1
fi

"$SEVENZIP" x "$ARCHIVE" "-o$ROOT_DIR" -y

echo "Features_mapped extracted successfully."
