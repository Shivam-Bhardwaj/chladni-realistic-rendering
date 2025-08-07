#!/bin/bash
# Emoji Detection Script for Linux/macOS
# Run this before committing to ensure no emojis are present

echo "Checking for emojis in project files..."
echo "====================================="

python3 .pre-commit-hooks/emoji-detector.py --check-all

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Emojis detected! Please remove them before committing."
    exit 1
else
    echo ""
    echo "SUCCESS: Repository is emoji-free!"
fi