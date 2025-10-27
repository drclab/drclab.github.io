#!/usr/bin/env bash
# Interactive helper to copy selected images from the Windows Downloads folder
# into the current working directory (or an optional destination argument).
set -euo pipefail

DEST_DIR="${1:-$PWD}"
SOURCE_BASE="/mnt/c/Users"

detect_windows_downloads() {
  if [[ -n "${WINDOWS_DOWNLOADS_DIR:-}" ]]; then
    printf '%s\n' "${WINDOWS_DOWNLOADS_DIR}"
    return 0
  fi

  if [[ -n "${WINDOWS_USER:-}" ]]; then
    printf '%s\n' "${SOURCE_BASE}/${WINDOWS_USER}/Downloads"
    return 0
  fi

  if [[ ! -d "${SOURCE_BASE}" ]]; then
    echo "Expected Windows users directory ${SOURCE_BASE} does not exist." >&2
    return 1
  fi

  mapfile -t USER_DIRS < <(find "${SOURCE_BASE}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
  declare -a CANDIDATES=()
  for user in "${USER_DIRS[@]}"; do
    if [[ -d "${SOURCE_BASE}/${user}/Downloads" ]]; then
      CANDIDATES+=("${user}")
    fi
  done

  if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
    echo "No Windows Downloads directories were found under ${SOURCE_BASE}." >&2
    echo "Set WINDOWS_USER or WINDOWS_DOWNLOADS_DIR to point to the correct location." >&2
    return 1
  fi

  if [[ "${#CANDIDATES[@]}" -eq 1 ]]; then
    printf '%s\n' "${SOURCE_BASE}/${CANDIDATES[0]}/Downloads"
    return 0
  fi

  echo "Multiple Windows user folders detected. Select one to use:"
  for idx in "${!CANDIDATES[@]}"; do
    printf '%3d) %s\n' "$((idx + 1))" "${CANDIDATES[$idx]}"
  done
  echo
  read -r -p "Enter the number for the Windows user: " user_choice
  if ! [[ "${user_choice}" =~ ^[0-9]+$ ]]; then
    echo "Invalid selection. Expected a number from the list." >&2
    return 1
  fi
  selected=$((user_choice - 1))
  if (( selected < 0 || selected >= ${#CANDIDATES[@]} )); then
    echo "Selection ${user_choice} is out of range." >&2
    return 1
  fi
  printf '%s\n' "${SOURCE_BASE}/${CANDIDATES[$selected]}/Downloads"
}

SOURCE_DIR="$(detect_windows_downloads)" || exit 1

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Expected downloads directory ${SOURCE_DIR} does not exist." >&2
  exit 1
fi

mapfile -t IMAGE_FILES < <(find "${SOURCE_DIR}" -maxdepth 1 -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.gif' \
     -o -iname '*.webp' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) \
  | sort)

if [[ "${#IMAGE_FILES[@]}" -eq 0 ]]; then
  echo "No image files found in ${SOURCE_DIR}." >&2
  exit 0
fi

echo "Select the images to copy from ${SOURCE_DIR}:"
for idx in "${!IMAGE_FILES[@]}"; do
  printf "%3d) %s\n" "$((idx + 1))" "$(basename "${IMAGE_FILES[$idx]}")"
done

echo
read -r -p "Enter the numbers to copy (space or comma separated): " SELECTION
if [[ -z "${SELECTION//[[:space:],]/}" ]]; then
  echo "No selection provided. Nothing to copy."
  exit 0
fi

IFS=', ' read -r -a CHOICES <<< "${SELECTION}"

declare -a TO_COPY
declare -A SEEN
for choice in "${CHOICES[@]}"; do
  if [[ -z "${choice}" ]]; then
    continue
  fi
  if ! [[ "${choice}" =~ ^[0-9]+$ ]]; then
    echo "Ignoring non-numeric selection: ${choice}"
    continue
  fi
  index=$((choice - 1))
  if (( index < 0 || index >= ${#IMAGE_FILES[@]} )); then
    echo "Ignoring out-of-range selection: ${choice}"
    continue
  fi
  if [[ -n "${SEEN[$index]:-}" ]]; then
    continue
  fi
  SEEN[$index]=1
  TO_COPY+=("${IMAGE_FILES[$index]}")
done

if [[ "${#TO_COPY[@]}" -eq 0 ]]; then
  echo "No valid selections. Nothing to copy."
  exit 0
fi

mkdir -p "${DEST_DIR}"

echo "Copying files to ${DEST_DIR}:"
for file in "${TO_COPY[@]}"; do
  base_name="$(basename "${file}")"
  dest_path="${DEST_DIR}/${base_name}"
  if [[ -e "${dest_path}" ]]; then
    echo " - Skipping ${base_name} (already exists)."
    continue
  fi
  cp -a "${file}" "${dest_path}"
  echo " - Copied ${base_name}"
done

echo "Done."
