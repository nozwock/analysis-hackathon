# Hackfest Project

## Using

- First, make sure the project root is the work directory.
- Using `pip`.
    - Create a virtual environment and activate it. Go to [Creating Virtual Environments](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments), to learn how.
    - Install package and its dependencies and run it with:
        ```console
        python -m pip install -e .
        analysis-hackfest
        ```
- Using `poertry`, install the project and run it with:
    ```console
    poetry install
    poetry run -- analysis-hackfest
    ```