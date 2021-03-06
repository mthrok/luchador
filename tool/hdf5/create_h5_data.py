#!/usr/bin/env python
"""Create HDF5 batch data using NumPy NDArray"""
import h5py

# pylint: disable=unused-import,eval-used
import numpy as np  # noqa


def _parse_command_line_args():
    import argparse
    ap = argparse.ArgumentParser(
        description=(
            'Create simple NumPy data and save it in HDF5. \n'
            'For example, '
            '\n\n'
            '    python {} "np.zeros((3, 4, 5))" foo.h5 --key bar'
            '\n\n'
            'creates HDF5 file called "foo.h5" which contains dataset called '
            '"bar", \nof which value is zero-array with shape (3, 4, 5).'
            .format(__file__)
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument('expression', help='Expression to create data.')
    ap.add_argument('output', help='Output file name.')
    ap.add_argument('--key', help='Name of dataset in HDF5', default='data')
    return ap.parse_args()


def _save(data, output_file, key='data'):
    file_ = h5py.File(output_file, 'a')
    if key in file_:
        del file_[key]
    file_.create_dataset(key, data=data)
    file_.close()


def _main():
    args = _parse_command_line_args()
    data = eval('{}'.format(args.expression))
    _save(data, args.output, args.key)


if __name__ == '__main__':
    _main()
