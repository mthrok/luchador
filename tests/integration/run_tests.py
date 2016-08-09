def run_integration_test(mode):
    if mode not in ['tensorflow', 'theano']:
        raise ValueError('`mode` must be either "tensorflow" or "theano"')

    cnn_format = 'NHWC' if mode == 'tensorflow' else 'NCHW'

    import os
    import time
    import numpy as np

    # import theano
    # theano.config.optimizer = 'None'
    # theano.config.exception_verbosity = 'high'

    import luchador
    luchador.set_nn_backend(mode)
    luchador.set_nn_conv_format(cnn_format)

    from luchador.nn import Session
    from luchador.nn import Input
    from luchador.nn import DeepQLearning
    from luchador.nn import SSE2
    from luchador.nn import GravesRMSProp
    from luchador.nn import SummaryWriter
    from luchador.nn.models import model_factory

    max_delta, min_delta = 1.0, -1.0
    learning_rate = 0.00025
    decay1, decay2 = 0.95, 0.95
    batchfile = 'mini-batch_Breakout-v0.npz'

    n_actions = 6
    discount_rate = 0.99
    height = width = 84
    history = 4
    batch = None
    state_shape = (
        (batch, height, width, history) if cnn_format == 'NHWC' else
        (batch, history, height, width))

    def model_maker():
        dqn = (
            model_factory('image_normalizer', denom=255) +
            model_factory('vanilla_dqn', n_actions=n_actions)
        )
        dqn(Input(shape=state_shape))
        return dqn

    print 'Building Q networks'
    ql = DeepQLearning(discount_rate)
    ql.build(model_maker)

    print 'Building Error'
    sse2 = SSE2(min_delta=min_delta, max_delta=max_delta)
    error = sse2(ql.target_q, ql.pre_trans_model.output)

    print 'Building Optimization'
    rmsprop = GravesRMSProp(
        learning_rate=learning_rate, decay1=decay1, decay2=decay2)
    params = ql.pre_trans_model.get_parameter_variables()
    minimize_op = rmsprop.minimize(error, wrt=params.values())

    print 'Initializing Session'
    session = Session()
    session.initialize()

    print 'Initializing SummaryWriter'
    outputs = ql.pre_trans_model.get_output_tensors()
    writer = SummaryWriter('./monitoring/test_tensorflow')
    writer.add_graph(session.graph)
    writer.register('pre_trans_network_params', 'histogram', params.keys())
    writer.register('pre_trans_network_outputs', 'histogram', outputs.keys())

    print 'Running computation'
    data = np.load(os.path.join(os.path.dirname(__file__), batchfile))
    pre_states = data['prestates']
    post_states = data['poststates']
    actions, rewards = data['actions'], data['rewards']
    terminals = data['terminals']
    if cnn_format == 'NHWC':
        pre_states = pre_states.transpose((0, 2, 3, 1))
        post_states = post_states.transpose((0, 2, 3, 1))

    sync_time = []
    summary_time = []
    run_time = []
    for i in range(100):
        if i % 10 == 0:
            print 'Syncing'
            t0 = time.time()
            session.run(name='sync', updates=ql.sync_op)
            sync_time.append(time.time() - t0)

            print 'Summarizing ', i
            t0 = time.time()
            params_vals = session.run(name='params', outputs=params.values())
            output_vals = session.run(
                name='outputs', outputs=outputs.values(), inputs={
                    ql.pre_states: pre_states
                })
            writer.summarize('pre_trans_network_params', i, params_vals)
            writer.summarize('pre_trans_network_outputs', i, output_vals)
            summary_time.append(time.time() - t0)

        t0 = time.time()
        e, target_qs, future_rewards, post_qs, pre_qs = session.run(
            name='minibatch',
            outputs=[
                error,
                ql.target_q,
                ql.future_reward,
                ql.post_trans_model.output,
                ql.pre_trans_model.output
            ],
            inputs={
                ql.pre_states: pre_states,
                ql.actions: actions,
                ql.rewards: rewards,
                ql.post_states: post_states,
                ql.terminals: terminals,
            },
            updates=minimize_op)
        """
        manual_error = 0.0
        for tgt_q, ftr_r, post_q, pre_q, action, reward in zip(
                target_qs, future_rewards, post_qs, pre_qs, actions, rewards):
            '''
            print 'Target Q', tgt_q
            print 'Pre Q   ', pre_q
            print 'Future R', ftr_r
            print 'Action  ', action
            print 'Reward  ', reward
            print 'Post Q  ', post_q
            print ''
            '''
            delta = tgt_q - pre_q
            delta = np.minimum(delta, max_delta)
            delta = np.maximum(delta, min_delta)
            manual_error += np.sum((delta ** 2) / 2)
        print 'Error:', e
        print 'Error:', manual_error / len(actions)
        print ''
        """
        run_time.append(time.time() - t0)

    print 'Sync   : {} ({})'.format(np.mean(sync_time), sync_time[0])
    print 'Summary: {} ({})'.format(np.mean(summary_time), summary_time[0])
    print 'Run    : {} ({})'.format(np.mean(run_time), run_time[0])
