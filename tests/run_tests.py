import os
import subprocess

# Get the current directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# List all files in the directory
all_files = os.listdir(current_dir)

# Filter out files that match the pattern 'test_*.py' and exclude 'test.py'
test_scripts = [f for f in all_files if f.startswith('test_')
                and f.endswith('.py') and f != 'test.py']

# Run each script
for script in test_scripts:
    script_path = os.path.join(current_dir, script)
    print(f"Running {script_path}...")
    result = subprocess.run(['python', script_path], capture_output=True,
                            text=True)
    print(f"Output of {script}:\n{result.stdout}")
    if result.stderr:
        print(f"Error in {script}:\n{result.stderr}")
