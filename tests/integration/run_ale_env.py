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
ap.add_argument('--frame_skip', type=int, default=4)
ap.add_argument('--random_start', type=int, default=0)
ap.add_argument('--grayscale',
                dest='grayscale', action='store_true')
ap.add_argument('--color',
                dest='grayscale', action='store_false')
ap.add_argument('--minimal_action_set',
                dest='minimal_action_set', action='store_true')
ap.add_argument('--legal_action_set',
                dest='minimal_action_set', action='store_false')
ap.add_argument('--mode', choices=['test', 'train'], default='train')
ap.add_argument('--plot', action='store_true')
args = ap.parse_args()

env = ALEEnvironment(
    args.rom,
    display_screen=args.display_screen, play_sound=args.sound,
    record_screen_path=args.record_screen_path,
    frame_skip=args.frame_skip,
    random_start=args.random_start,
    grayscale=args.grayscale,
    minimal_action_set=args.minimal_action_set,
    mode=args.mode)

logger.info('\n{}'.format(env))

n_actions = env.n_actions
for episode in range(10):
    total_reward = 0.0
    env.reset()
    ep_frame0 = env.ale.getEpisodeFrameNumber()
    for n_steps in range(1, 10000):
        a = np.random.randint(n_actions)
        reward, screen, terminal, info = env.step(a)
        total_reward += reward
        if terminal:
            break

    ep_frame1 = info['episode_frame_number']
    logger.info('Episode {}:'.format(episode))
    logger.info('  Score : {}'.format(total_reward))
    logger.info('  Lives : {}'.format(info['lives']))
    logger.info('  #Steps: {}'.format(n_steps))
    logger.info('  #EpisodeFrame: {} -> {}'.format(ep_frame0, ep_frame1))
    logger.info('  #Total Frames: {}'.format(info['total_frame_number']))

logger.info('Screen type: {}'.format(type(screen)))
logger.info('Screen shape: {}'.format(screen.shape))


if args.plot:
    import matplotlib.pyplot as plt
    plt.imshow(screen, cmap='Greys_r')
    plt.show()
