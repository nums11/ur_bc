#!/usr/bin/env python3
import requests
import time
import numpy as np
import math
import json
from rich.console import Console
from rich.table import Table
import argparse

# Reference joint positions [deg]: [0, -90, 90, -90, -90, 0]
# Convert to radians for calculations
REFERENCE_POSITION_DEG = [0, -90, 90, -90, -90, 0]
REFERENCE_POSITION_RAD = [math.radians(angle) for angle in REFERENCE_POSITION_DEG]

def fetch_data(url):
    """Fetch data from the specified URL."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def calculate_offsets(joint_positions, reference_positions):
    """Calculate offsets between current joint positions and reference positions."""
    if not isinstance(joint_positions, list) or len(joint_positions) != 6:
        print(f"Error: Expected list of 6 joint positions, got {joint_positions}")
        return None
    
    # Calculate offsets (current - reference)
    offsets = [current - reference for current, reference in zip(joint_positions, reference_positions)]
    return offsets

def normalize_joint_positions(joint_positions, normalize_method="auto"):
    """
    Normalize joint positions if they appear to be in a different unit than radians.
    
    Args:
        joint_positions: The joint positions to normalize
        normalize_method: 
            "auto": Automatically detect and normalize values that appear to be in degrees
            "degrees": Convert from degrees to radians
            "none": Don't normalize
    """
    if normalize_method == "none":
        return joint_positions
    
    normalized = []
    for i, pos in enumerate(joint_positions):
        # If values are too large to be radians, they're likely in degrees
        if normalize_method == "auto" and abs(pos) > 10:  # Most joint values in radians are between -π and π
            normalized.append(math.radians(pos))
        elif normalize_method == "degrees":
            normalized.append(math.radians(pos))
        else:
            normalized.append(pos)
    
    return normalized

def extract_joint_positions(data):
    """Extract the 6 joint positions from the data received from the endpoint."""
    # This function needs to be adjusted based on the actual data format
    # For now, we assume the data contains joint positions directly or in a nested structure
    
    if isinstance(data, dict) and "joint_positions" in data:
        # If data has a "joint_positions" key
        return data["joint_positions"]
    elif isinstance(data, dict) and all(f"q{i}" in data for i in range(6)):
        # If data has q0, q1, ..., q5 keys
        return [data[f"q{i}"] for i in range(6)]
    elif isinstance(data, list) and len(data) >= 6:
        # If data is a list with at least 6 elements
        return data[:6]
    else:
        # Try to find joint positions in the data structure
        print("Could not automatically extract joint positions. Please check the data format.")
        print(f"Received data: {data}")
        return None

def display_data(joint_positions, normalized_positions, offsets_deg, offsets_rad):
    """Display the joint positions and offsets in a formatted table."""
    if not joint_positions or not offsets_deg or not offsets_rad:
        print("No data to display")
        return
    
    console = Console()
    
    # Create a table
    table = Table(title="Joint Positions and Offsets")
    
    # Add columns
    table.add_column("Joint", style="cyan")
    table.add_column("Raw Position", style="white")
    table.add_column("Normalized (rad)", style="green")
    table.add_column("Reference (rad)", style="blue")
    table.add_column("Offset (rad)", style="yellow")
    table.add_column("Offset (deg)", style="magenta")
    
    # Add rows for each joint
    for i in range(6):
        table.add_row(
            f"Joint {i}",
            f"{joint_positions[i]:.6f}",
            f"{normalized_positions[i]:.6f}",
            f"{REFERENCE_POSITION_RAD[i]:.6f}",
            f"{offsets_rad[i]:.6f}",
            f"{offsets_deg[i]:.2f}"
        )
    
    console.print(table)

def save_offsets_to_file(joint_positions, normalized_positions, offsets_rad, offsets_deg, filename="joint_offsets.json"):
    """Save the joint positions and offsets to a JSON file."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reference_position_deg": REFERENCE_POSITION_DEG,
        "reference_position_rad": REFERENCE_POSITION_RAD,
        "raw_position": joint_positions,
        "normalized_position_rad": normalized_positions,
        "offsets_rad": offsets_rad,
        "offsets_deg": offsets_deg
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"\nJoint positions and offsets saved to {filename}")

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate joint offsets between current positions and reference.")
    parser.add_argument("--url", default="http://10.19.2.209/", help="URL of the endpoint to fetch joint positions")
    parser.add_argument("--normalize", choices=["auto", "degrees", "none"], default="auto", 
                        help="How to normalize the joint positions (auto, degrees, none)")
    parser.add_argument("--output", default="joint_offsets.json", help="Output file for the offsets")
    return parser.parse_args()

def main():
    args = parse_args()
    
    url = args.url
    normalize_method = args.normalize
    output_file = args.output
    
    print(f"Fetching data from {url}")
    print(f"Reference position (deg): {REFERENCE_POSITION_DEG}")
    print(f"Reference position (rad): {[f'{angle:.6f}' for angle in REFERENCE_POSITION_RAD]}")
    print(f"Normalization method: {normalize_method}")
    
    try:
        # Fetch data once
        data = fetch_data(url)
        if data:
            joint_positions = extract_joint_positions(data)
            
            if joint_positions:
                # Normalize joint positions if needed
                normalized_positions = normalize_joint_positions(joint_positions, normalize_method)
                
                # Calculate offsets in radians
                offsets_rad = calculate_offsets(normalized_positions, REFERENCE_POSITION_RAD)
                
                # Convert offsets to degrees for display
                offsets_deg = [math.degrees(offset) for offset in offsets_rad]
                
                print("\n" + "="*70)
                print(f"Data retrieved at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                display_data(joint_positions, normalized_positions, offsets_deg, offsets_rad)
                
                # Save to file
                save_offsets_to_file(joint_positions, normalized_positions, offsets_rad, offsets_deg, output_file)
            else:
                print("Could not extract joint positions from data")
        else:
            print("Failed to fetch data from endpoint")
    except KeyboardInterrupt:
        print("\nProgram terminated by user")

if __name__ == "__main__":
    main() 