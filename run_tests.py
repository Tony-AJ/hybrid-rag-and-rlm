import subprocess
import sys
import os

def run_tests():
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        
        result = subprocess.run(
            ["poetry", "run", "pytest", "tests/test_api_compliance.py", "-v"],
            capture_output=True,
            text=True,
            env=env
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        with open("test_output.txt", "w") as f:
            f.write(result.stdout)
            f.write("\nSTDERR:\n")
            f.write(result.stderr)
            
    except Exception as e:
        print(f"Error: {e}")
        with open("test_output.txt", "w") as f:
            f.write(f"Error: {e}")

if __name__ == "__main__":
    run_tests()
