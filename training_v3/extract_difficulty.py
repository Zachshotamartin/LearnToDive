"""Extract the degree-of-difficulty tables from the World Aquatics rulebook PDF."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pymupdf

SOURCE = 'https://resources.fina.org/fina/document/2026/02/18/e6815ecc-06d9-4f0b-98e9-4c441cf5e6a3/2026-02-18_World-Aquatics_CR-Final.pdf'
VERSION = 'World Aquatics February 2026, appendices 9 and 11'
# Zero-based PDF page ranges of the two tables and the heights each column block covers.
TABLES = [('springboard', range(163, 166), [1, 3]), ('platform', range(168, 173), [10, 7.5, 5])]
CODE = re.compile(r'[1-6]\d{2,3}')
VALUE = re.compile(r'\d\.\d')


def merge_split_name(cells, heights):
    """Some rows split the dive name over three cells; join them back into one."""
    if len(cells) == 4 + 4 * len(heights):
        return [cells[0], ' '.join(x for x in cells[1:4] if x)] + cells[4:]
    return cells


def collect(document):
    rows = {}
    for apparatus, pages, heights in TABLES:
        for page in pages:
            for table in document[page].find_tables().tables:
                for cells in table.extract():
                    cells = merge_split_name(cells, heights)
                    if not cells[0] or not CODE.fullmatch(cells[0].strip()) or len(cells) != 2 + 4 * len(heights):
                        continue
                    record_row(rows, cells, apparatus, heights)
    return rows


def record_row(rows, cells, apparatus, heights):
    code = cells[0].strip()
    for j, cell in enumerate(cells[2:]):
        if not cell or not VALUE.fullmatch(cell.strip()):
            continue
        position = 'ABCD'[j % 4]
        key = f'{code}{position}'
        record = rows.setdefault(key, dict(id=key, code=int(code), position=position, name=cells[1].replace('\n', ' '),
                                           difficulty={}))
        column = f'{apparatus}:{heights[j // 4]:g}'
        value = float(cell)
        if column in record['difficulty'] and record['difficulty'][column] != value:
            raise ValueError((key, column))
        record['difficulty'][column] = value


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument('pdf')
    args = arguments.parse_args()
    rows = collect(pymupdf.open(args.pdf))
    catalog = dict(source=SOURCE, sourceSHA256=hashlib.sha256(Path(args.pdf).read_bytes()).hexdigest(), version=VERSION,
                   rows=sorted(rows.values(), key=lambda x: x['id']))
    (Path(__file__).resolve().parent / 'difficulty.json').write_text(json.dumps(catalog, indent=2) + '\n')
    for key in ['101C', '103B', '203C', '5132D', '6243D']:
        print(key, rows.get(key))
    print(len(rows), 'dive/position combinations', sum(len(x['difficulty']) for x in rows.values()), 'DD values')


if __name__ == '__main__':
    main()
