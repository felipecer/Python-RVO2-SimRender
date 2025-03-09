import os
import shutil
import argparse

def remove_logs_and_saves(base_folder: str):
    """
    Recursively scan base_folder for subdirectories named 'logs' or 'saves'
    and delete them (and their contents).
    """
    for root, dirs, _ in os.walk(base_folder):
        if 'logs' in dirs:
            logs_path = os.path.join(root, 'logs')
            shutil.rmtree(logs_path, ignore_errors=True)
        if 'saves' in dirs:
            saves_path = os.path.join(root, 'saves')
            shutil.rmtree(saves_path, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="Delete 'logs' and 'saves' directories in subfolders.")
    parser.add_argument("base_folder", type=str, help="Path to the base folder to scan")
    args = parser.parse_args()
    remove_logs_and_saves(args.base_folder)

if __name__ == "__main__":
    main()