#!/usr/bin/env python3
"""Extract CS2 demo files from RAR archives.

This script scans RAR archives for CS2 demo files (.dem), prioritizing
de_dust2 maps, and extracts them for testing.
"""

import subprocess
import re
from pathlib import Path
from typing import List, Tuple


def list_rar_contents(rar_path: Path) -> List[str]:
    """List files in RAR archive.

    Args:
        rar_path: Path to RAR archive

    Returns:
        List of filenames in the archive
    """
    try:
        result = subprocess.run(
            ["unrar", "l", str(rar_path)],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse unrar output to find .dem files
        lines = result.stdout.split('\n')
        dem_files = []

        for line in lines:
            # Look for lines containing .dem files
            if '.dem' in line.lower():
                # Extract filename from the line
                # unrar output format: " <size> <date> <time>  <filename>"
                parts = line.split()
                if parts:
                    # Filename is usually the last part
                    filename = parts[-1]
                    if filename.endswith('.dem'):
                        dem_files.append(filename)

        return dem_files

    except subprocess.CalledProcessError as e:
        print(f"  Error listing {rar_path.name}: {e}")
        return []


def extract_demo(rar_path: Path, demo_name: str, output_dir: Path) -> bool:
    """Extract specific demo from RAR archive.

    Args:
        rar_path: Path to RAR archive
        demo_name: Name of demo file to extract
        output_dir: Directory to extract to

    Returns:
        True if extraction successful, False otherwise
    """
    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract with overwrite (-o+)
        subprocess.run(
            ["unrar", "e", "-o+", str(rar_path), demo_name, str(output_dir)],
            capture_output=True,
            check=True
        )
        return True

    except subprocess.CalledProcessError as e:
        print(f"  Error extracting {demo_name}: {e}")
        return False


def find_dust2_demos(data_dir: Path, limit: int = 5) -> List[Tuple[Path, str]]:
    """Scan RAR archives and find de_dust2 demos.

    Args:
        data_dir: Directory containing RAR archives
        limit: Maximum number of demos to find

    Returns:
        List of (rar_path, demo_filename) tuples
    """
    found = []

    print(f"Scanning RAR archives in {data_dir}...")

    rar_files = list(data_dir.glob("*.rar"))
    print(f"Found {len(rar_files)} RAR archives")

    for i, rar_file in enumerate(rar_files, 1):
        if len(found) >= limit:
            break

        print(f"  [{i}/{len(rar_files)}] Checking {rar_file.name}...", end='')

        contents = list_rar_contents(rar_file)

        if not contents:
            print(" no .dem files found")
            continue

        # Find dust2 demos
        dust2_demos = [f for f in contents if 'dust2' in f.lower()]

        if dust2_demos:
            print(f" found {len(dust2_demos)} dust2 demo(s)")
            for demo in dust2_demos:
                found.append((rar_file, demo))
                if len(found) >= limit:
                    break
        else:
            print(f" {len(contents)} demo(s), but no dust2")

    return found


def get_file_size(file_path: Path) -> str:
    """Get human-readable file size.

    Args:
        file_path: Path to file

    Returns:
        Size string (e.g., "45.2 MB")
    """
    size_bytes = file_path.stat().st_size

    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def main():
    """Main extraction workflow."""
    import argparse, os
    parser = argparse.ArgumentParser(description="Extract sample CS2 demos from HLTV RAR archives.")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get("HLTV_DATA_DIR", "../misc_cs/hltv-scrape/data")),
                        help="Directory containing HLTV .rar archives.")
    parser.add_argument("--dust2-output-dir", type=Path, default=Path("test_data/real_demos/de_dust2"))
    parser.add_argument("--other-output-dir", type=Path, default=Path("test_data/real_demos/other_maps"))
    args = parser.parse_args()
    data_dir = args.data_dir
    dust2_output_dir = args.dust2_output_dir
    other_output_dir = args.other_output_dir

    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    print("=" * 70)
    print("CS2 Demo Extraction Script")
    print("=" * 70)
    print()

    # Find dust2 demos
    print("Step 1: Finding de_dust2 demos...")
    print("-" * 70)
    dust2_demos = find_dust2_demos(data_dir, limit=5)

    if not dust2_demos:
        print("\n⚠️  No de_dust2 demos found!")
        print("Falling back to extracting any demos for manual inspection...")
        # Could implement fallback here
        return

    print(f"\n✓ Found {len(dust2_demos)} de_dust2 demo(s)")
    print()

    # Extract dust2 demos
    print("Step 2: Extracting de_dust2 demos...")
    print("-" * 70)
    extracted_count = 0

    for rar_path, demo_name in dust2_demos[:2]:  # Limit to 2 demos as planned
        print(f"\nExtracting: {demo_name}")
        print(f"  From: {rar_path.name}")

        if extract_demo(rar_path, demo_name, dust2_output_dir):
            extracted_file = dust2_output_dir / Path(demo_name).name
            if extracted_file.exists():
                size = get_file_size(extracted_file)
                print(f"  ✓ Extracted: {size}")
                extracted_count += 1
            else:
                print(f"  ✗ File not found after extraction")
        else:
            print(f"  ✗ Extraction failed")

    print()
    print("=" * 70)
    print("Extraction Summary")
    print("=" * 70)
    print(f"Demos extracted: {extracted_count}")
    print(f"Output directory: {dust2_output_dir.absolute()}")
    print()

    # List extracted files
    if extracted_count > 0:
        print("Extracted files:")
        for dem_file in dust2_output_dir.glob("*.dem"):
            print(f"  - {dem_file.name} ({get_file_size(dem_file)})")
    else:
        print("⚠️  No demos were successfully extracted")

    print()
    print("Next step: Run 'python scripts/analyze_real_demo.py' to analyze demos")


if __name__ == "__main__":
    main()
