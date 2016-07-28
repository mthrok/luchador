def run_integration_test(mode):
    if mode not in ['tensorflow', 'theano']:
        raise ValueError('`mode` must be either "tensorflow" or "theano"')

    cnn_format = 'NHWC' if mode == 'tensorflow' else 'NCHW'

    import time
    import numpy as np
    # import theano
    # theano.config.optimizer = 'None'
    # theano.config.exception_verbosity = 'high'
    import luchador
    luchador.set_nn_backend(mode)
    import luchador.nn
    luchador.nn.set_cnn_format(cnn_format)

    from luchador.nn import (
        Session, Input, DeepQLearning, SSE, RMSProp, SummaryWriter)
    from luchador.nn.models import model_factory

    discount_rate = 0.99
    n_actions = 16
    height = width = 84
    channel = 4
    batch = 32
    state_shape = ((batch, height, width, channel) if cnn_format == 'NHWC'
                   else (batch, channel, height, width))

    def model_maker():
        dqn = model_factory('vanilla_dqn', n_actions=n_actions)
        dqn(Input(shape=state_shape))
        return dqn

    print 'Building Q networks'
    qli = DeepQLearning(discount_rate)
    qli.build(model_maker)

    print 'Building Error'
    sse = SSE(min_delta=1.01, max_delta=1.02)
    error = sse(qli.target_q, qli.pre_trans_model.output)

    print 'Building Optimization'
    rmsprop = RMSProp(0.1)
    params = qli.pre_trans_model.get_parameter_variables()
    minimize_op = rmsprop.minimize(error, wrt=params.values())

    print 'Initializing Session'
    session = Session()
    session.initialize()

    print 'Initializing SummaryWriter'
    outputs = qli.pre_trans_model.get_output_tensors()
    writer = SummaryWriter('./monitoring/test_tensorflow')
    writer.add_graph(session.graph)
    writer.register('pre_trans_network_params', 'histogram', params)
    writer.register('pre_trans_network_outputs', 'histogram', outputs)

    print 'Running computation'
    pre_states = np.ones(state_shape, dtype=np.uint8)
    post_states = np.ones(state_shape, dtype=np.uint8)
    actions = np.random.randint(0, n_actions, (batch,), dtype=np.uint8)
    rewards = np.ones((batch,), dtype=np.float32)
    terminals = np.ones((batch,), dtype=np.bool)

    sync_time = []
    summary_time = []
    run_time = []
    for i in range(100):
        if i % 10 == 0:
            print 'Syncing'
            t0 = time.time()
            session.run(name='sync', updates=qli.sync_op)
            sync_time.append(time.time() - t0)

            print 'Summarizing ', i
            t0 = time.time()
            params_vals = session.run(name='params', outputs=params.values())
            output_vals = session.run(
                name='outputs', outputs=outputs.values(), inputs={
                    qli.pre_states: pre_states
                })
            writer.summarize('pre_trans_network_params', i, params_vals)
            writer.summarize('pre_trans_network_outputs', i, output_vals)
            summary_time.append(time.time() - t0)

        t0 = time.time()
        session.run(name='minibatch', outputs=qli.target_q, inputs={
            qli.pre_states: pre_states,
            qli.actions: actions,
            qli.rewards: rewards,
            qli.post_states: post_states,
            qli.terminals: terminals,
        }, updates=minimize_op)
        run_time.append(time.time() - t0)

    print 'Sync   : {} ({})'.format(np.mean(sync_time), sync_time[0])
    print 'Summary: {} ({})'.format(np.mean(summary_time), summary_time[0])
    print 'Run    : {} ({})'.format(np.mean(run_time), run_time[0])
