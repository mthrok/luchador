import logging
from argparse import ArgumentParser as AP

import numpy as np

from luchador.env import ALEEnvironment


logger = logging.getLogger('luchador')

ap = AP()
ap.add_argument('--rom', default='breakout')
ap.add_argument('--display_screen', '-screen', action='store_true')
ap.add_argument('--sound', action='store_true')
ap.add_argument('--record_screen_path')
ap.add_argument('--grayscale', type=bool, default=True)
ap.add_argument('--frame_skip', type=int, default=4)
ap.add_argument('--minimal_action_set',
                dest='minimal_action_set', action='store_true')
ap.add_argument('--legal_action_set',
                dest='minimal_action_set', action='store_false')
ap.add_argument('--mode', choices=['test', 'train'])
args = ap.parse_args()

env = ALEEnvironment(
    args.rom,
    display_screen=args.display_screen, sound=args.sound,
    record_screen_path=args.record_screen_path,
    frame_skip=args.frame_skip,
    grayscale=args.grayscale,
    minimal_action_set=args.minimal_action_set,
    mode=args.mode)

logger.info('\n{}'.format(env))

n_actions = env.n_actions
for episode in range(10):
    total_reward = 0.0
    terminal = False
    while not terminal:
        a = np.random.randint(n_actions)
        reward, screen, terminal, info = env.step(a)
        total_reward += reward
    env.reset()
    logger.info('Episode {}:'.format(episode))
    logger.info('  Score: {}, Lives: {}'.format(total_reward, info['lives']))
    logger.info('  Frame Number: {} / {}'
                .format(info['episode_frame_number'], info['frame_number']))

logger.info('Screen type: {}'.format(type(screen)))
logger.info('Screen shape: {}'.format(screen.shape))


'''
import matplotlib.pyplot as plt
if args.grayscale:
    plt.imshow(screen[:, :, 0], cmap='Greys_r')
else:
    plt.imshow(screen)
plt.show()
'''
