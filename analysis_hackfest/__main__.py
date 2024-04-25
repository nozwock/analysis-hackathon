import subprocess
from pathlib import Path


def run():
    try:
        subprocess.run(["streamlit", "run", "Home.py"], cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        ...


if __name__ == "__main__":
    run()
