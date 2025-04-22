import os

def cat_all_files(directory):
    """Reads and prints out all file contents from a directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            # Only cat .py or text-based files
            if file.endswith('.py') or file.endswith('.txt') or file.endswith('.md'):
                print(f"\n===== {file_path} =====\n")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(content)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    # Path to your pinball_ai directory
    directory = "pinball_ai"
    
    if os.path.exists(directory):
        cat_all_files(directory)
    else:
        print(f"Directory {directory} does not exist.")