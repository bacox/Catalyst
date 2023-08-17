from pathlib import Path

if __name__ == '__main__':

    def to_int(name: str):
        return int(''.join(filter(str.isdigit, name.split('_')[0].lstrip('exp'))))
    
    print('Available experiments:')
    script_dir = Path(__file__).parent
    for file_name in sorted([x.stem for x in script_dir.iterdir() if x.suffix == '.py' and not x.stem.startswith('__')], key=lambda x:to_int(x)):
        print(file_name)