from __future__ import absolute_import

import os
import csv
import logging

from luchador.common import load_config
from luchador.nn import (
    Saver,
    Session,
    get_optimizer,
)
import formula

_LG = logging.getLogger('luchador')


def parse_command_line_args():
    from argparse import ArgumentParser as AP
    ap = AP(
        description='Run optimizer on simple formular and save the result'
    )
    ap.add_argument(
        'formula',
        help='Name of formula on which optimizer is run.'
    )
    ap.add_argument(
        'optimizer',
        help='Optimizer configuration to test'
    )
    ap.add_argument(
        '--output',
        help='File path to save result.'
    )
    return ap.parse_args()


def get_formula(name):
    try:
        return getattr(formula, name).get()
    except Exception:
        _LG.exception('Formula ({}) not found.'.format(name))
        raise


def serialize(optimizer, loss, wrt):
    optimizer.minimize(loss=loss, wrt=wrt)

    sess = Session()
    sess.initialize()
    _LG.info('Serializing Optimizer.')
    params = optimizer.get_parameter_variables()
    values = sess.run(outputs=params.values())
    return {key: value for key, value in zip(params.keys(), values)}


def load_optimizer(name):
    if not name.endswith('.yml'):
        name = '{}.yml'.format(name)
    path = os.path.join(os.path.dirname(__file__), 'optimizer', name)
    cfg = load_config(path)
    Optimizer = get_optimizer(cfg['name'])
    return Optimizer(**cfg['args'])


def save_result(filepath, result):
    directory = os.path.dirname(os.path.abspath(filepath))
    if not os.path.exists(directory):
        os.makedirs(directory)

    # TODO: Limit data points
    with open(filepath, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=result[0].keys())
        writer.writeheader()
        for r in result:
            writer.writerow(r)


def main():
    args = parse_command_line_args()
    _LG.info('Running {} on {}'.format(args.optimizer, args.formula))
    formula = get_formula(args.formula)
    optimizer = load_optimizer(args.optimizer)
    result = serialize(
        optimizer=optimizer, loss=formula['loss'], wrt=formula['wrt'])
    if args.output:
        _LG.info('Saving at {}'.format(args.output))
        saver = Saver(args.output)
        saver.save(result)

if __name__ == '__main__':
    main()
