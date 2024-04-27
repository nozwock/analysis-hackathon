import subprocess
import sys
from pathlib import Path


def run():
    try:
        subprocess.run(
            ["streamlit", "run", "streamlit_app.py", *sys.argv[1:]],
            cwd=Path(__file__).parent,
        )
    except KeyboardInterrupt:
        ...


if __name__ == "__main__":
    run()
