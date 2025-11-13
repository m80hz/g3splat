#!/usr/bin/env bash
# download_scannet_v1_test_meshes.sh
# Usage:
#   ./download_scannet_v1_test_meshes.sh /path/to/scannet_test_pairs.txt [OUT_ROOT]
#
# Notes:
# - Optional: set COOKIE_FILE=/path/to/cookies.txt if the server requires login cookies.

set -euo pipefail
IFS=$'\n\t'

LIST_FILE="${1:-scannetv1_test/scannet_test_pairs.txt}"
OUT_ROOT="${2:-scannetv1_test}"
BASE_URL="${BASE_URL:-http://kaldir.vc.in.tum.de/scannet/v1/scans}"
# If set to 1, use sudo only for creating dirs and moving final files
USE_SUDO_SAVE="${USE_SUDO_SAVE:-0}"

if [[ "$USE_SUDO_SAVE" == "1" ]]; then
  # cache sudo once so you don't get prompted for every file
  sudo -v || { echo "sudo auth failed"; exit 1; }
fi

ensure_dir() {
  local d="$1"
  if [[ "$USE_SUDO_SAVE" == "1" ]]; then sudo mkdir -p "$d"; else mkdir -p "$d"; fi
}

save_file() {
  local src="$1" dest="$2"
  if [[ "$USE_SUDO_SAVE" == "1" ]]; then
    sudo mv "$src" "$dest"
    sudo chmod 644 "$dest" || true
  else
    mv "$src" "$dest"
  fi
}


if [[ ! -f "$LIST_FILE" ]]; then
  echo "Input list '$LIST_FILE' not found." >&2
  exit 1
fi


# Choose downloader
if command -v aria2c >/dev/null 2>&1; then
  DL="aria2c"
elif command -v wget >/dev/null 2>&1; then
  DL="wget"
elif command -v curl >/dev/null 2>&1; then
  DL="curl"
else
  echo "Please install aria2c or wget or curl." >&2
  exit 1
fi

# Extract unique scene IDs from the first column
mapfile -t SCENES < <(awk '{print $1}' "$LIST_FILE" \
  | sed 's/\r$//' \
  | grep -E '^scene[0-9]+_[0-9]+$' \
  | sort -u)

echo "Found ${#SCENES[@]} unique scene IDs."

download_one() {
  local id="$1"
  local url="$BASE_URL/$id/${id}_vh_clean_2.ply"
  local dest_dir="$OUT_ROOT/$id/mesh"
  local dest="$dest_dir/${id}_vh_clean_2.ply"

  # create dest dir (sudo if requested)
  ensure_dir "$dest_dir"

  # skip if already present
  if [[ -s "$dest" ]]; then
    echo "[skip] $dest (already exists)"
    return 0
  fi

  # download to a user-writable temp file, then sudo-move into place
  local tmp
  tmp="$(mktemp)" || { echo "mktemp failed"; return 1; }
  trap 'rm -f "$tmp"' RETURN

  echo "[get ] $id → $dest"
  case "$DL" in
    aria2c)
      if [[ -n "${COOKIE_FILE:-}" && -f "$COOKIE_FILE" ]]; then
        aria2c -x8 -s8 -c --load-cookies="$COOKIE_FILE" \
          -d "$(dirname "$tmp")" -o "$(basename "$tmp")" "$url"
      else
        aria2c -x8 -s8 -c \
          -d "$(dirname "$tmp")" -o "$(basename "$tmp")" "$url"
      fi
      ;;
    wget)
      if [[ -n "${COOKIE_FILE:-}" && -f "$COOKIE_FILE" ]]; then
        wget --load-cookies="$COOKIE_FILE" -c -O "$tmp" "$url"
      else
        wget -c -O "$tmp" "$url"
      fi
      ;;
    curl)
      if [[ -n "${COOKIE_FILE:-}" && -f "$COOKIE_FILE" ]]; then
        curl -L -C - -b "$COOKIE_FILE" -o "$tmp" "$url"
      else
        curl -L -C - -o "$tmp" "$url"
      fi
      ;;
  esac

  # move into the protected location
  save_file "$tmp" "$dest"
  trap - RETURN
  rm -f "$tmp" 2>/dev/null || true
}


for id in "${SCENES[@]}"; do
  download_one "$id"
done

echo "All done ✅"
