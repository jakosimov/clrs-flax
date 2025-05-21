import marimo

__generated_with = "0.13.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import clrs
    import numpy as np
    import jax
    import jax.numpy as jnp
    import pprint
    return clrs, jax, np, pprint


@app.cell
def _(jax, np):
    rng = np.random.RandomState(1234)
    rng_key = jax.random.PRNGKey(rng.randint(2**32))
    return (rng,)


@app.cell
def _(clrs, pprint):
    # algorithm = 'bubble_sort'
    algorithm = "matrix_chain_order"
    n_nodes = 8
    batch_size = 8

    train_sampler, spec = clrs.build_sampler(
        name=algorithm, num_samples=100, length=n_nodes
    )

    test_sampler, spec = clrs.build_sampler(
        name=algorithm, num_samples=100, length=n_nodes * 4
    )

    pprint.pprint(spec)


    def _iterate_sampler(sampler, batch_size):
        while True:
            yield sampler.next(batch_size)


    train_sampler = _iterate_sampler(
        train_sampler, batch_size=batch_size
    )
    test_sampler = _iterate_sampler(
        test_sampler, batch_size=100
    )
    return spec, test_sampler, train_sampler


@app.cell
def _(np, train_sampler):
    dummy_trajectory = next(train_sampler)


    def print_datapoint(probe, is_hint=False):
        print(probe)
        if probe.name == "pos":
            return
        if is_hint:
            data = probe.data[3]
        else:
            data = probe.data
        if probe.location == "node":
            print("values in data type")
            print(np.unique(data[0]))
        elif probe.location == "edge":
            print("values in data type")
            print(np.unique(data[0]))


    def print_trajectory(trajectory):
        inputs = trajectory.features.inputs
        hints = trajectory.features.hints
        outputs = trajectory.outputs
        print()
        print("== inputs")
        for input in inputs:
            # (batch, node)
            print_datapoint(input)
        print()

        print("== hints")
        for hint in hints:
            # (step, batch, node)
            print_datapoint(hint, is_hint=True)
        print()
        print("== outputs")
        for output in outputs:
            print_datapoint(output)


    print("Dummy trajectory:")
    print_trajectory(dummy_trajectory)
    return (dummy_trajectory,)


@app.cell
def _():
    # gat_processor_factory = clrs.get_processor_factory('gat', use_ln=True, nb_triplet_fts=32, nb_heads=4)
    # gat_model_params = dict(
    #     processor_factory=gat_processor_factory,
    #     hidden_dim=32,
    #     encode_hints=True,
    #     decode_hints=True,
    #     # decode_diffs=False,
    #     # hint_teacher_forcing_noise=1.0,
    #     use_lstm=False,
    #     learning_rate=0.001,
    #     checkpoint_path='/tmp/checkpt',
    #     freeze_processor=False,
    #     dropout_prob=0.0,
    # )

    # gat_model = clrs.models.BaselineModel(
    #     spec=spec,
    #     dummy_trajectory=dummy_trajectory,
    #     **gat_model_params
    # )

    # gat_model.init(dummy_trajectory.features, 1234)
    return


@app.cell
def _():
    from jakobs import processors
    from jakobs import baselines
    from flax import nnx
    import jax.random as random

    mpnn_processor_factory = processors.get_processor_factory(
        processors.ProcessorKind.MPNN,
        use_ln=True,
        nb_triplet_fts=32,
        nb_heads=4,
        reduction=processors.AggregationMode.MAX,
    )

    mpnn_model_params = dict(
        processor_factory=mpnn_processor_factory,
        hidden_dim=32,
        encode_hints=True,
        decode_hints=True,
        use_lstm=False,
        checkpoint_path="/tmp/checkpt",
        freeze_processor=False,
        dropout_prob=0.0,
    )
    return baselines, mpnn_model_params, nnx, random


@app.cell
def _(clrs, jax, nnx, rng):
    def train_model(
        model,
        train_sampler,
        test_sampler,
        optimizer,
        max_steps=200,
    ):
        rng_key = jax.random.PRNGKey(rng.randint(2**32))
        step = 0
        while step <= max_steps:
            feedback, test_feedback = (
                next(train_sampler),
                next(test_sampler),
            )
            rng_key, new_rng_key = jax.random.split(rng_key)

            def loss_fn(model):
                return model.feedback(rng_key, feedback)

            cur_loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(grads)
            rng_key = new_rng_key
            if step % 10 == 0:
                predictions_val, _ = model.predict(
                    rng_key, feedback.features
                )
                out_val = clrs.evaluate(
                    feedback.outputs, predictions_val
                )
                predictions, _ = model.predict(
                    rng_key, test_feedback.features
                )
                out = clrs.evaluate(
                    test_feedback.outputs, predictions
                )
                print(
                    f"step = {step} | loss = {cur_loss} | val_acc = {out_val['score']} | test_acc = {out['score']}"
                )
            step += 1

        return model
    return (train_model,)


@app.cell
def _():
    import optax

    learning_rate = 1e-2

    grad_clip_max_norm = 0.0
    if grad_clip_max_norm != 0.0:
        optax_chain = [
            optax.clip_by_global_norm(grad_clip_max_norm),
            optax.scale_by_adam(),
            optax.scale(-learning_rate),
        ]
        opt = optax.chain(*optax_chain)
    else:
        opt = optax.adam(learning_rate)
    return learning_rate, opt


@app.cell
def _(
    baselines,
    dummy_trajectory,
    mpnn_model_params,
    nnx,
    opt,
    random,
    spec,
    test_sampler,
    train_model,
    train_sampler,
):
    rngs = nnx.Rngs(params=0, dropout=random.key(1))
    mpnn_model = baselines.BaselineModel(
        spec=spec,
        dummy_trajectory=dummy_trajectory,
        rngs=rngs,
        **mpnn_model_params,
    )

    for path, module in mpnn_model.iter_modules():
        print(path, type(module).__name__)

    optimizer = nnx.Optimizer(mpnn_model, opt)
    train_model(
        mpnn_model,
        train_sampler,
        test_sampler,
        optimizer,
        max_steps=100,
    )
    return


@app.cell
def _(clrs, dummy_trajectory, learning_rate, spec):
    old_mpnn_processor_factory = clrs.get_processor_factory(
        "mpnn", use_ln=True, nb_triplet_fts=32, nb_heads=4
    )
    old_mpnn_model_params = dict(
        processor_factory=old_mpnn_processor_factory,
        hidden_dim=32,
        encode_hints=True,
        decode_hints=True,
        use_lstm=False,
        learning_rate=learning_rate,
        checkpoint_path="/tmp/checkpt",
        freeze_processor=False,
        dropout_prob=0.0,
    )

    old_mpnn_model = clrs.models.BaselineModel(
        spec=spec,
        dummy_trajectory=dummy_trajectory,
        **old_mpnn_model_params,
    )

    old_mpnn_model.init(dummy_trajectory.features, 1234)
    return (old_mpnn_model,)


@app.cell
def _(clrs, jax, rng):
    def old_train_model(
        model, train_sampler, test_sampler, max_steps=200
    ):
        rng_key = jax.random.PRNGKey(rng.randint(2**32))
        step = 0
        while step <= max_steps:
            feedback, test_feedback = (
                next(train_sampler),
                next(test_sampler),
            )
            # print('trajectory')
            # print_trajectory(feedback)
            rng_key, new_rng_key = jax.random.split(rng_key)
            cur_loss = model.feedback(rng_key, feedback)
            rng_key = new_rng_key
            if step % 10 == 0:
                predictions_val, _ = model.predict(
                    rng_key, feedback.features
                )
                out_val = clrs.evaluate(
                    feedback.outputs, predictions_val
                )
                predictions, _ = model.predict(
                    rng_key, test_feedback.features
                )
                out = clrs.evaluate(
                    test_feedback.outputs, predictions
                )
                print(
                    f"step = {step} | loss = {cur_loss} | val_acc = {out_val['score']} | test_acc = {out['score']}"
                )
            step += 1

        return model
    return (old_train_model,)


@app.cell
def _(old_mpnn_model, old_train_model, test_sampler, train_sampler):
    old_train_model(
        old_mpnn_model,
        train_sampler,
        test_sampler,
        max_steps=100,
    )
    return


if __name__ == "__main__":
    app.run()
