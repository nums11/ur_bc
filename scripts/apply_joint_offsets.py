#!/usr/bin/env python3
import requests
import time
import json
import math
import numpy as np
import argparse
from rich.console import Console
from rich.table import Table

def fetch_data(url):
    """Fetch data from the specified URL."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def extract_joint_positions(data):
    """Extract the 6 joint positions from the data received from the endpoint."""
    if isinstance(data, dict) and "joint_positions" in data:
        return data["joint_positions"]
    elif isinstance(data, dict) and all(f"q{i}" in data for i in range(6)):
        return [data[f"q{i}"] for i in range(6)]
    elif isinstance(data, list) and len(data) >= 6:
        return data[:6]
    else:
        print("Could not automatically extract joint positions. Please check the data format.")
        print(f"Received data: {data}")
        return None

def load_offsets(filename):
    """Load joint offsets from a JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading offsets file: {e}")
        return None

def apply_offsets(joint_positions, offset_data, normalize_method="auto"):
    """
    Apply offsets to joint positions to match the reference position.
    
    Args:
        joint_positions: The current joint positions
        offset_data: The offset data loaded from the file
        normalize_method: How to normalize the joint positions
    
    Returns:
        The joint positions to send to the UR robot
    """
    # Normalize the joint positions if needed
    normalized = []
    for i, pos in enumerate(joint_positions):
        if normalize_method == "auto" and abs(pos) > 10:
            normalized.append(math.radians(pos))
        elif normalize_method == "degrees":
            normalized.append(math.radians(pos))
        else:
            normalized.append(pos)
    
    # Get the reference positions from the offset data
    reference_positions = offset_data["reference_position_rad"]
    
    # Calculate the offsets between the current and reference positions
    offsets = [ref - curr for curr, ref in zip(normalized, reference_positions)]
    
    # Apply the offsets to get the UR robot joint positions
    # These are the positions needed to move the UR robot to match the reference position
    ur_positions = reference_positions
    
    return ur_positions, normalized, offsets

def display_data(joint_positions, normalized_positions, ur_positions, offsets):
    """Display the joint positions and calculated UR positions in a formatted table."""
    console = Console()
    
    # Create a table
    table = Table(title="Joint Positions and UR Robot Commands")
    
    # Add columns
    table.add_column("Joint", style="cyan")
    table.add_column("Raw Position", style="white")
    table.add_column("Normalized (rad)", style="green")
    table.add_column("Offset (rad)", style="yellow")
    table.add_column("UR Position (rad)", style="magenta")
    table.add_column("UR Position (deg)", style="blue")
    
    # Add rows for each joint
    for i in range(6):
        table.add_row(
            f"Joint {i}",
            f"{joint_positions[i]:.6f}",
            f"{normalized_positions[i]:.6f}",
            f"{offsets[i]:.6f}",
            f"{ur_positions[i]:.6f}",
            f"{math.degrees(ur_positions[i]):.2f}"
        )
    
    console.print(table)
    
    # Also print the UR robot joint array for easy copy-paste
    print("\nUR Robot Joint Positions (rad):")
    print(f"[{', '.join([f'{pos:.6f}' for pos in ur_positions])}]")
    
    print("\nUR Robot Joint Positions (deg):")
    print(f"[{', '.join([f'{math.degrees(pos):.2f}' for pos in ur_positions])}]")

def parse_args():
    parser = argparse.ArgumentParser(description="Apply joint offsets to control UR robot.")
    parser.add_argument("--url", default="http://10.19.2.209/", help="URL of the endpoint to fetch joint positions")
    parser.add_argument("--offsets", default="joint_offsets.json", help="JSON file containing joint offsets")
    parser.add_argument("--normalize", choices=["auto", "degrees", "none"], default="auto", 
                        help="How to normalize the joint positions (auto, degrees, none)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    url = args.url
    offsets_file = args.offsets
    normalize_method = args.normalize
    
    print(f"Fetching data from {url}")
    print(f"Using offsets from {offsets_file}")
    print(f"Normalization method: {normalize_method}")
    
    # Load the offsets
    offset_data = load_offsets(offsets_file)
    if not offset_data:
        print("Failed to load offsets data. Run joint_offset_calculator.py first.")
        return
    
    try:
        # Fetch data once
        data = fetch_data(url)
        if data:
            joint_positions = extract_joint_positions(data)
            
            if joint_positions:
                # Apply offsets to get UR robot positions
                ur_positions, normalized, offsets = apply_offsets(joint_positions, offset_data, normalize_method)
                
                print("\n" + "="*70)
                print(f"Data retrieved at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                display_data(joint_positions, normalized, ur_positions, offsets)
            else:
                print("Could not extract joint positions from data")
        else:
            print("Failed to fetch data from endpoint")
    except KeyboardInterrupt:
        print("\nProgram terminated by user")

if __name__ == "__main__":
    main() 