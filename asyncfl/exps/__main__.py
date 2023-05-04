from pathlib import Path

if __name__ == '__main__':
    print('Available experiments:')
    script_dir = Path(__file__).parent
    for file_name in [x.stem for x in script_dir.iterdir() if x.suffix == '.py' and not x.stem.startswith('__')]:
        print(file_name)