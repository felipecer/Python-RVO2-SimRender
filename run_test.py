import os
import unittest

def discover_tests(start_dir, pattern):
    test_dirs = []
    exclude_dirs = {'__pycache__'}  # Set of directories to exclude

    for root, dirs, files in os.walk(start_dir):
        # Exclude unwanted directories by modifying the 'dirs' list in-place
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        # Check each file in the directory
        for file in files:
            if file.startswith(pattern):
                test_dirs.append(root)
                break  # If a matching file is found, add the directory and stop searching this directory

    return test_dirs

def run_tests():
    test_root = './tests/'
    pattern = 'test_'  # Ensure this pattern exactly matches the start of your test files
    test_dirs = discover_tests(test_root, pattern)
    
    print(f"Test directories found: {test_dirs}")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all tests from each discovered directory
    for test_dir in test_dirs:
        suite.addTests(loader.discover(start_dir=test_dir, pattern='test*.py'))
    
    print(f"Total tests found: {suite.countTestCases()}.")

    if suite.countTestCases() == 0:
        print("No tests found. Check the path and file pattern.")
        return
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result

if __name__ == "__main__":
    run_tests()
