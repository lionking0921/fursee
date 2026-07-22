import os
import sys
import shutil

def clear_folder(relative_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_folder = os.path.join(current_dir, relative_path)
    
    if not os.path.exists(target_folder):
        print(f"[SKIPPED] Folder '{relative_path}' not found in the same directory.")
        return
    if not os.path.isdir(target_folder):
        print(f"[SKIPPED] '{relative_path}' is a file, not a folder.")
        return
        
    print(f"Clearing contents of '{relative_path}'...")
    for item_name in os.listdir(target_folder):
        item_path = os.path.join(target_folder, item_name)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
            # print(f"  [SUCCESS] Deleted: {item_name}")
        except Exception as e:
            print(f"  [FAILED] Failed to delete: {item_name} | Reason: {e}")
    print(f"Folder '{relative_path}' has been cleared successfully!\n")

FOLDER_MAP = {
    "output": [
        "output/classify",
        "output/identify",
        "output/similar",
    ],
    "buffer": [
        "buffer",
    ],
    "all": [
        "buffer",
        "input/id_targets",
        "input/images",
        "input/sim_targets",
        "output/classify",
        "output/identify",
        "output/similar",
    ]
}

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Error: Missing required argument.")
        print(f"Usage: python {os.path.basename(__file__)} <scope>")
        print(f"Available scopes: {', '.join(FOLDER_MAP.keys())}")
        sys.exit(1)

    scope = sys.argv[1].lower()
    
    if scope not in FOLDER_MAP:
        print(f"Error: Invalid scope '{scope}'.")
        print(f"Available scopes: {', '.join(FOLDER_MAP.keys())}")
        sys.exit(1)

    folders_to_clear = FOLDER_MAP[scope]
    print(f"Starting reset with scope: '{scope}'\n")
    
    for folder in folders_to_clear:
        clear_folder(folder)
        
    print("Reset process completed.")