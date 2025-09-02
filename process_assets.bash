#!/bin/bash
set -euo pipefail

CONFIG_FILE="characters.json"
MEDIA_DIR="media"
IMAGES_DIR="images"
MAX_DURATION=300
SAMPLE_RATE=24000
TEMP_DIR="/tmp/asset_processor_$$"

for cmd in curl ffmpeg jq bc; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "Error: Missing required command: '$cmd'. Please install it." >&2
        exit 1
    fi
done

[ ! -f "$CONFIG_FILE" ] && { echo "Error: '$CONFIG_FILE' not found."; exit 1; }

trap 'rm -rf "$TEMP_DIR"' EXIT

rm -rf "$MEDIA_DIR" "$IMAGES_DIR"
mkdir -p "$MEDIA_DIR" "$IMAGES_DIR" "$TEMP_DIR"

jq -c '.[]' "$CONFIG_FILE" | while IFS= read -r character; do
    slug=$(jq -r '.slug // empty' <<< "$character")
    media_src=$(jq -r '.voicesnippet // empty' <<< "$character")
    image_src=$(jq -r '.image // empty' <<< "$character")
    ranges=$(jq -c '.voiceranges // empty' <<< "$character")

    if [[ -z "$slug" ]]; then
        echo "Warning: Skipping character with missing slug."
        continue
    fi

    if [[ -n "$media_src" && "$media_src" != "null" ]]; then
        output_file="${MEDIA_DIR}/${slug}.wav"

        if [[ -n "$ranges" && "$ranges" != "[]" && "$ranges" != "null" ]]; then
            accumulated_duration=0
            idx=1
            concat_list_file="${TEMP_DIR}/${slug}_concat.txt"
            rm -f "$concat_list_file"

            while IFS= read -r range; do
                if (( $(echo "$accumulated_duration >= $MAX_DURATION" | bc -l) )); then
                    break
                fi

                start=$(jq -r '.[0]' <<< "$range")
                end=$(jq -r '.[1]' <<< "$range")
                
                if (( $(echo "$end > $start" | bc -l) )); then
                    range_duration=$(echo "$end - $start" | bc)
                    remaining_time=$(echo "$MAX_DURATION - $accumulated_duration" | bc)
                    
                    if (( $(echo "$range_duration > $remaining_time" | bc -l) )); then
                        duration_to_extract=$remaining_time
                    else
                        duration_to_extract=$range_duration
                    fi
                    
                    part_file="${TEMP_DIR}/${slug}_part${idx}.wav"
                    ffmpeg -y -ss "$start" -i "$media_src" -t "$duration_to_extract" \
                        -vn -acodec pcm_s16le -ar "$SAMPLE_RATE" -ac 1 \
                        -loglevel error "$part_file" < /dev/null
                    
                    echo "file '$(realpath "$part_file")'" >> "$concat_list_file"
                    ((idx++))
                    accumulated_duration=$(echo "$accumulated_duration + $duration_to_extract" | bc)
                fi
            done < <(jq -c '.[]' <<< "$ranges")

            if [[ -f "$concat_list_file" ]]; then
                ffmpeg -y -f concat -safe 0 -i "$concat_list_file" -c copy "$output_file" < /dev/null 2>/dev/null
            fi
        else
            ffmpeg -y -i "$media_src" -t "$MAX_DURATION" \
                -vn -acodec pcm_s16le -ar "$SAMPLE_RATE" -ac 1 \
                -loglevel error "$output_file" < /dev/null
        fi
    fi

    if [[ -n "$image_src" && "$image_src" != "null" ]]; then
        final_image="${IMAGES_DIR}/${slug}.jpg"
        if [[ "$image_src" == http* ]]; then
            curl -sfL --connect-timeout 10 -o "$final_image" "$image_src"
        else
            cp "$image_src" "$final_image"
        fi
    fi
done

