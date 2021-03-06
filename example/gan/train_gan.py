"""Train vanilla GAN on MNIST"""
from __future__ import division
from __future__ import absolute_import

import os
import logging

import numpy as np
from luchador import nn

from example.utils import (
    initialize_logger, load_mnist, plot_images
)

_LG = logging.getLogger(__name__)


def _parse_command_line_args():
    import argparse
    default_mnist_path = os.path.join(
        os.path.expanduser('~'), '.dataset', 'mnist.pkl.gz')
    default_model_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'gan.yml'
    )
    parser = argparse.ArgumentParser(
        description=(
            'Train simple Generative Adversarial Network '
            'on MNIST dataset.'
        )
    )
    parser.add_argument(
        '--model', default=default_model_file,
        help=(
            'Model configuration file. '
            'Default: {}'.format(default_model_file)
        )
    )
    parser.add_argument(
        '--n-iterations', default=1000, type=int,
        help='#optimizatitons par epoch.'
    )
    parser.add_argument(
        '--n-epochs', default=10, type=int,
        help='#Epochs to run.'
    )
    parser.add_argument(
        '--n-seeds', default=100, type=int,
        help='#Generator input dimensions.'
    )
    parser.add_argument(
        '--dataset', default=default_mnist_path,
        help=(
            'Path to MNIST dataset, downloaded from '
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz '
            'Default: {}'.format(default_mnist_path)
        ),
    )
    parser.add_argument(
        '--output',
        help=(
            'When provided, plot generated data to this directory.'
        )
    )
    parser.add_argument(
        '--mock', action='store_true',
        help='Mock test data to run the script without data for testing.'
    )
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def _build_models(model_file):
    _LG.info('Loading model %s', model_file)
    model_def = nn.get_model_config(model_file)
    return nn.make_model(model_def)


def _build_loss(logit_real, logit_fake):
    sce_gen = nn.cost.SigmoidCrossEntropy(scope='sce_gen')
    sce_real = nn.cost.SigmoidCrossEntropy(scope='sce_real')
    sce_fake = nn.cost.SigmoidCrossEntropy(scope='sce_fake')
    gen_loss = sce_gen(prediction=logit_fake, target=1)

    disc_loss_real = sce_real(prediction=logit_real, target=1)
    disc_loss_fake = sce_fake(prediction=logit_fake, target=0)
    disc_loss = disc_loss_real + disc_loss_fake
    return gen_loss, disc_loss


def _build_optimization(generator, gen_loss, discriminator, disc_loss):
    optimizer_disc = nn.optimizer.Adam(
        learning_rate=0.001, scope='TrainDiscriminator/Adam')
    optimizer_gen = nn.optimizer.Adam(
        learning_rate=0.001, scope='TrainGenerator/Adam')

    opt_gen = optimizer_gen.minimize(
        gen_loss, generator.get_parameters_to_train())
    opt_disc = optimizer_disc.minimize(
        disc_loss, discriminator.get_parameters_to_train())
    return opt_gen, opt_disc


def _train(
        train_disc, train_gen, plot_samples,
        n_iterations, n_epochs):
    plot_samples(0)
    _LG.info('%5s: %10s, %10s', 'EPOCH', 'DISC_LOSS', 'GEN_LOSS')
    for epoch in range(1, n_epochs+1):
        disc_loss, gen_loss = 0.0, 0.0
        for _ in range(n_iterations):
            disc_loss += train_disc() / n_iterations
            gen_loss += train_gen() / n_iterations
        _LG.info('%5s: %10.3e, %10.3e', epoch, disc_loss, gen_loss)
        plot_samples(epoch)


def _sample_seed(*size):
    return np.random.uniform(-1., 1., size=size).astype('float32')


def _main():
    args = _parse_command_line_args()
    initialize_logger(args.debug)

    batch_size = 32
    dataset = load_mnist(args.dataset, flatten=True, mock=args.mock)

    model = _build_models(args.model)
    discriminator, generator = model['discriminator'], model['generator']

    input_gen = nn.Input(shape=(None, args.n_seeds), name='GeneratorInput')
    data_real = nn.Input(shape=dataset.train.shape, name='InputData')
    data_fake = generator(input_gen)

    logit_fake = discriminator(data_fake)
    logit_real = discriminator(data_real)

    gen_loss, disc_loss = _build_loss(logit_real, logit_fake)
    opt_gen, opt_disc = _build_optimization(
        generator, gen_loss, discriminator, disc_loss)

    sess = nn.Session()
    sess.initialize()

    if args.output:
        summary = nn.SummaryWriter(output_dir=args.output)
        if sess.graph is not None:
            summary.add_graph(sess.graph)

    def _train_disc():
        return sess.run(
            inputs={
                input_gen: _sample_seed(batch_size, args.n_seeds),
                data_real: dataset.train.next_batch(batch_size).data
            },
            outputs=disc_loss,
            updates=opt_disc,
            name='train_discriminator',
        )

    def _train_gen():
        return sess.run(
            inputs={
                input_gen: _sample_seed(batch_size, args.n_seeds),
            },
            outputs=gen_loss,
            updates=opt_gen,
            name='train_generator',
        )

    def _plot_samples(epoch):
        if not args.output:
            return
        images = sess.run(
            inputs={
                input_gen: _sample_seed(16, args.n_seeds),
            },
            outputs=data_fake,
            name='generate_samples',
        ).reshape(-1, 28, 28)
        path = os.path.join(args.output, '{:03d}.png'.format(epoch))
        plot_images(images, path)

    _train(
        _train_disc, _train_gen, _plot_samples,
        args.n_iterations, args.n_epochs
    )


if __name__ == '__main__':
    _main()
