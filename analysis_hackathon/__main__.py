import subprocess
from pathlib import Path

if __name__ == "__main__":
    try:
        subprocess.run(["streamlit", "run", "Home.py"], cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        ...
