import jax
import jax.numpy as jnp
import numpy as np

from iqp_cond.exact_tvd_training import (
    CampaignTrainingConfig,
    sample_nonidentity_masks,
    shuffled_batches,
    train_campaign,
)


def test_shuffled_batches_are_200_complete_epochs():
    config = CampaignTrainingConfig(steps=2000, batch_size=100)
    batches = shuffled_batches(3, 1000, config, 7)
    assert batches.shape == (2000, 3, 100)
    for condition in range(3):
        for epoch in (0, 199):
            block = batches[epoch * 10 : (epoch + 1) * 10, condition].ravel()
            assert np.array_equal(np.sort(block), np.arange(1000))


def test_nonidentity_masks_have_exact_analytic_normalization():
    n_qubits, sigma = 8, 1.3
    masks, weights = sample_nonidentity_masks(
        jax.random.PRNGKey(3), 2000, n_qubits, sigma
    )
    p = (1.0 - np.exp(-1.0 / (2.0 * sigma**2))) / 2.0
    assert np.all(np.asarray(masks).sum(axis=1) > 0)
    assert len(np.unique(np.asarray(masks), axis=0)) == len(masks)
    assert np.isclose(float(weights.sum()), 1.0 - (1.0 - p) ** n_qubits)


def test_partial_epoch_preserves_full_epoch_schedule_and_budget():
    partial = shuffled_batches(2, 300, CampaignTrainingConfig(steps=500), 4711)
    complete = shuffled_batches(2, 300, CampaignTrainingConfig(steps=501), 4711)
    np.testing.assert_array_equal(partial, complete[:500])
    assert partial.shape == (500, 2, 100)
    for condition in range(2):
        for start in range(0, 498, 3):
            np.testing.assert_array_equal(
                np.sort(partial[start : start + 3, condition].ravel()), np.arange(300)
            )
        assert len(np.unique(partial[498:, condition])) == 200


def test_complete_epoch_schedule_matches_original_rng_order():
    rng = np.random.default_rng(7)
    expected = np.empty((20, 3, 100), dtype=np.int32)
    for start in (0, 10):
        for condition in range(3):
            expected[start : start + 10, condition] = rng.permutation(1000).reshape(
                10, 100
            )
    actual = shuffled_batches(3, 1000, CampaignTrainingConfig(steps=20), 7)
    np.testing.assert_array_equal(actual, expected)


def test_schedule_free_checkpoints_use_evaluation_parameters():
    config = CampaignTrainingConfig(
        steps=10,
        batch_size=5,
        validation_interval=5,
        optimizer="schedule_free_adamw",
        learning_rate=0.01,
        warmup_steps=2,
    )
    targets = np.zeros((1, 10, 1), dtype=np.float64)

    def loss(params, _batch, _key):
        return jnp.sum((params - 1.0) ** 2)

    def validation(params):
        return jnp.sum((params - 1.0) ** 2)

    result = train_campaign(np.zeros(2), targets, validation, loss, 2, config)
    final_checkpoint = next(
        checkpoint
        for checkpoint in result.checkpoints
        if checkpoint.step == 10 and checkpoint.reason == "scheduled"
    )
    assert np.all(np.isfinite(final_checkpoint.params))
    assert np.isclose(
        final_checkpoint.validation_mmd, validation(final_checkpoint.params)
    )
