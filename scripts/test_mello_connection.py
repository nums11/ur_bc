#!/usr/bin/env python3
import requests
import time
import json
from rich.console import Console
from rich.table import Table

def fetch_data(url):
    """Fetch data from the specified URL."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()  # Assuming the response is in JSON format
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def display_data(data):
    """Display the data in a formatted table."""
    if not data:
        print("No data to display")
        return
    
    console = Console()
    
    # Create a table
    table = Table(title="Data from HTTP Endpoint")
    
    # Add columns based on the data structure
    if isinstance(data, dict):
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in data.items():
            table.add_row(str(key), str(value))
    elif isinstance(data, list):
        # Handle list data differently if needed
        for i, item in enumerate(data):
            if isinstance(item, dict):
                # First item, create columns
                if i == 0:
                    for key in item.keys():
                        table.add_column(str(key), style="cyan")
                
                # Add row for each item
                table.add_row(*[str(v) for v in item.values()])
            else:
                # Simple list of values
                if i == 0:
                    table.add_column("Index", style="cyan")
                    table.add_column("Value", style="green")
                table.add_row(str(i), str(item))
    else:
        # Simple value
        table.add_column("Value", style="green")
        table.add_row(str(data))
    
    console.print(table)

def main():
    url = "http://10.19.2.209/"
    print(f"Fetching data from {url}")
    
    try:
        while True:
            data = fetch_data(url)
            if data:
                print("\n" + "="*50)
                print(f"Data retrieved at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                display_data(data)
            time.sleep(0.01)  # Poll 100 times per second (100 Hz)
    except KeyboardInterrupt:
        print("\nProgram terminated by user")

if __name__ == "__main__":
    main() 