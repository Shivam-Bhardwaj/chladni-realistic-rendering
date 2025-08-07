#!/usr/bin/env python3
"""
Emoji Detection and Prevention System
Prevents emoji characters from being committed to the repository.
"""

import sys
import re
import os
from typing import List, Tuple

# Comprehensive emoji detection pattern (excludes box drawing characters)
EMOJI_PATTERN = re.compile(
    r'[\U0001F600-\U0001F64F]|'  # emoticons
    r'[\U0001F300-\U0001F5FF]|'  # symbols & pictographs
    r'[\U0001F680-\U0001F6FF]|'  # transport & map
    r'[\U0001F1E0-\U0001F1FF]|'  # flags (iOS)
    r'[\U0001F900-\U0001F9FF]|'  # supplemental symbols
    r'[\U0001FA70-\U0001FAFF]|'  # symbols and pictographs extended-A
    r'[\U00002600-\U000026FF]|'  # miscellaneous symbols (excluding box drawing)
    r'[\U00002700-\U000027B0]'   # dingbats (excluding box drawing 0x2500-0x257F)
)

def find_emojis_in_file(filepath: str) -> List[Tuple[int, str, str]]:
    """
    Find all emojis in a file and return their line numbers and content.
    
    Returns:
        List of tuples: (line_number, line_content, emoji_found)
    """
    emojis_found = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                matches = EMOJI_PATTERN.findall(line)
                if matches:
                    for emoji in matches:
                        emojis_found.append((line_num, line.strip(), emoji))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []
    
    return emojis_found

def check_files_for_emojis(file_paths: List[str]) -> bool:
    """
    Check multiple files for emoji content.
    
    Returns:
        True if emojis found, False if clean
    """
    emojis_found = False
    
    # File extensions to check
    text_extensions = {'.md', '.txt', '.py', '.cpp', '.cu', '.h', '.hpp', '.c', '.js', '.html', '.css', '.yml', '.yaml', '.json'}
    
    for filepath in file_paths:
        if not os.path.exists(filepath):
            continue
            
        # Only check text files
        _, ext = os.path.splitext(filepath)
        if ext.lower() not in text_extensions:
            continue
            
        file_emojis = find_emojis_in_file(filepath)
        
        if file_emojis:
            emojis_found = True
            print(f"\\nEMOJI DETECTED in {filepath}:")
            for line_num, line_content, emoji in file_emojis:
                # Safely display emoji information without printing the actual emoji
                emoji_hex = f"U+{ord(emoji):04X}" if len(emoji) == 1 else f"[MULTI-CHAR-EMOJI]"
                print(f"  Line {line_num}: Found {emoji_hex} in: {line_content[:80].encode('ascii', 'replace').decode('ascii')}...")
    
    return emojis_found

def main():
    """Main emoji detection function."""
    print("Emoji Detection System - Professional Mode")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage: python emoji-detector.py <file1> <file2> ...")
        print("       python emoji-detector.py --check-all")
        sys.exit(1)
    
    if sys.argv[1] == "--check-all":
        # Check common documentation files
        files_to_check = [
            "README.md",
            "CHANGELOG.md", 
            "CONTRIBUTING.md",
            "FEATURES_v1.2.0.md",
            "PERFORMANCE_GUIDE.md",
            "src/main.cpp",
            "src/particle_physics.cu"
        ]
        
        # Add any other .md files in the directory
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.md') and file not in files_to_check:
                    files_to_check.append(os.path.join(root, file))
    else:
        files_to_check = sys.argv[1:]
    
    has_emojis = check_files_for_emojis(files_to_check)
    
    if has_emojis:
        print("\\nEMOJI DETECTION FAILED!")
        print("This repository must remain emoji-free for professional presentation.")
        print("Please remove all emoji characters before committing.")
        print("\\nTip: Use simple text alternatives like:")
        print("  RELEASE: or NEW: instead of rocket emoji")
        print("  FIX: or BUGFIX: instead of bug emoji") 
        print("  DONE: or COMPLETED: instead of check emoji")
        print("  TECHNICAL: or IMPROVEMENT: instead of wrench emoji")
        sys.exit(1)
    else:
        print("SUCCESS: All files are emoji-free. Professional standards maintained!")
        sys.exit(0)

if __name__ == "__main__":
    main()