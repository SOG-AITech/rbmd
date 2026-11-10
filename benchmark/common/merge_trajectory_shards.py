#!/usr/bin/env python3
"""Merge same-run MPI shards frame by frame for OVITO; memory is one selected frame.

Example: python3 merge_trajectory_shards.py -o window.lammpstrj rbmd.rank*.trj
"""
import argparse
from contextlib import ExitStack
from pathlib import Path


def read_frame(stream):
    first = stream.readline()
    if not first:
        return None
    if first != 'ITEM: TIMESTEP\n':
        raise ValueError(f'{stream.name}: invalid timestep header')
    step = int(stream.readline())
    if stream.readline() != 'ITEM: NUMBER OF ATOMS\n':
        raise ValueError(f'{stream.name}: invalid atom count header')
    count = int(stream.readline())
    if count < 0:
        raise ValueError('negative atom count')
    box = [stream.readline() for _ in range(4)]
    if not box[0].startswith('ITEM: BOX BOUNDS ') or any(not line for line in box):
        raise ValueError(f'{stream.name}: incomplete box header')
    header = stream.readline()
    if not header.startswith('ITEM: ATOMS '):
        raise ValueError(f'{stream.name}: invalid atom header')
    fields = header.split()[2:]
    atoms = []
    for _ in range(count):
        line = stream.readline()
        if len(line.split()) != len(fields):
            raise ValueError(f'{stream.name}: truncated or malformed atom row')
        atoms.append(line if line.endswith('\n') else line + '\n')
    return step, box, header, fields, atoms


def merge(paths, output):
    resolved = [p.resolve() for p in paths]
    if len(set(resolved)) != len(resolved) or output.resolve() in resolved:
        raise ValueError('duplicate input or output overlaps an input')
    frames = 0
    with ExitStack() as stack:
        streams = [stack.enter_context(p.open()) for p in paths]
        # Refuse accidental overwrite of an earlier merged result.
        out = stack.enter_context(output.open('x'))
        while True:
            group = [read_frame(s) for s in streams]
            if all(f is None for f in group):
                break
            if any(f is None for f in group):
                raise ValueError('shards have different frame counts')
            step, box, header, fields, _ = group[0]
            if any(f[:4] != group[0][:4] for f in group):
                raise ValueError('shards disagree on timestep, box or fields')
            rows = [row for f in group for row in f[4]]
            if 'id' in fields:
                index = fields.index('id')
                rows.sort(key=lambda row: int(row.split()[index]))
                ids = [int(row.split()[index]) for row in rows]
                if any(a == b for a, b in zip(ids, ids[1:])):
                    raise ValueError('duplicate atom IDs across shards')
            out.write(f'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(rows)}\n')
            out.writelines(box)
            out.write(header)
            out.writelines(rows)
            frames += 1
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('shards', nargs='+', type=Path)
    args = parser.parse_args()
    print(f'Merged {merge(args.shards, args.output)} frames into {args.output}')


if __name__ == '__main__':
    main()
