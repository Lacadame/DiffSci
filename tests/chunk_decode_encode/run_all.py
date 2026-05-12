"""Run all chunk_decode_encode tests in order."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    files = sorted(
        f for f in os.listdir(HERE)
        if f.startswith('test_') and f.endswith('.py')
    )
    print(f"Running {len(files)} test files in {HERE}")
    failed = []
    for fname in files:
        path = os.path.join(HERE, fname)
        print(f"\n==> {fname}")
        rc = subprocess.call([sys.executable, path])
        if rc != 0:
            failed.append(fname)
            print(f"    [FAIL] exit code {rc}")
    print()
    print('=' * 70)
    if failed:
        print(f"FAILED ({len(failed)}/{len(files)}):")
        for f in failed:
            print(f"  - {f}")
        return 1
    print(f"ALL {len(files)} PASSED")
    return 0


if __name__ == '__main__':
    sys.exit(main())
