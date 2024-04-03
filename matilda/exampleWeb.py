import sys

from buildIS import buildIS

def exampleWeb(rootdir: str) -> None:
    try:
        buildIS(rootdir)
    except Exception as e:
        print("EOF:ERROR")
        raise e

if __name__ == "__main__":
    rootdir = sys[-1]
    exampleWeb(rootdir)

    