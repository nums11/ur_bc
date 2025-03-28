#!/usr/bin/env python3
import serial
import time
import sys
from typing import Optional

def setup_serial(port: str = '/dev/serial/by-id/usb-M5Stack_Technology_Co.__Ltd_M5Stack_UiFlow_2.0_24587ce945900000-if00',
                baudrate: int = 115200) -> Optional[serial.Serial]:
    """
    Set up serial connection to M5Stack device.
    
    Args:
        port: Serial port path
        baudrate: Communication speed
        
    Returns:
        Serial object if successful, None otherwise
    """
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=1  # 1 second timeout
        )
        print(f"Successfully connected to {port}")
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return None

def read_serial_data(ser: serial.Serial) -> None:
    """
    Continuously read and print data from serial port.
    
    Args:
        ser: Serial object
    """
    try:
        while True:
            if ser.in_waiting:
                # Read line and decode
                line = ser.readline().decode('utf-8').strip()
                print(f"Received here: {line}")
            time.sleep(0.004)  # Small delay to prevent CPU overuse
            
    except KeyboardInterrupt:
        print("\nStopping serial read...")
    except Exception as e:
        print(f"Error reading serial data: {e}")
    finally:
        print("Closing serial port")
        ser.close()

def main():
    # Set up serial connection
    ser = setup_serial()
    if ser is None:
        sys.exit(1)
    
    try:
        # Give the device time to initialize
        time.sleep(2)
        
        # Start reading data
        print("Starting to read serial data. Press Ctrl+C to stop.")
        read_serial_data(ser)
        
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if ser.is_open:
            ser.close()
            print("Serial port closed")

if __name__ == "__main__":
    main() 