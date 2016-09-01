import logging
from argparse import ArgumentParser as AP

import numpy as np
# import matplotlib.pyplot as plt

from luchador.env import ALEEnvironment


logger = logging.getLogger('luchador')

ap = AP()
ap.add_argument('--rom', default='breakout')
ap.add_argument('--display_screen', '-screen', action='store_true')
ap.add_argument('--sound', action='store_true')
ap.add_argument('--record_screen_path')
ap.add_argument('--frame_skip', default=1, type=int)
args = ap.parse_args()

env = ALEEnvironment(
    args.rom,
    display_screen=args.display_screen, sound=args.sound,
    record_screen_path=args.record_screen_path,
    frame_skip=args.frame_skip)

logger.info('\n{}'.format(env))

n_actions = env.n_actions
for episode in range(10):
    total_reward = 0.0
    terminal = False
    while not terminal:
        a = np.random.randint(n_actions)
        reward, screen, terminal, info = env.step(a)
        total_reward += reward
    print type(screen)
    print screen.shape
    if screen.shape[2] == 1:
        screen = screen[:, :, 0]
    # plt.imshow(screen, cmap='Greys_r')
    # plt.show()

    frame_number = env.ale.getFrameNumber()
    frame_number_ep = env.ale.getEpisodeFrameNumber()
    env.reset()

    logger.info('Episode {}: Score: {}'.format(episode, total_reward))
    logger.info('Frame Number: {} / {}'.format(frame_number_ep, frame_number))
