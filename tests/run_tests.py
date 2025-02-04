import os
import sys
import subprocess


def run_tests():
    """Run all test files in the tests directory."""
    # Get absolute path to tests directory
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(tests_dir)

    # Add project root to Python path
    sys.path.insert(0, project_root)

    # Find all test files
    test_files = [f for f in os.listdir(tests_dir)
                  if f.startswith('test_') and f.endswith('.py')
                  and f != 'run_tests.py']

    print(f"Found test files: {', '.join(test_files)}\n")

    # Run each test file
    failed = False
    for test_file in test_files:
        print(f"\nRunning {test_file}...")
        result = subprocess.run([sys.executable, os.path.join(tests_dir, test_file)],
                                capture_output=True, text=True)

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Check result
        if result.returncode != 0:
            failed = True
            print(f"{test_file} failed with exit code {result.returncode}")

    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(run_tests())
