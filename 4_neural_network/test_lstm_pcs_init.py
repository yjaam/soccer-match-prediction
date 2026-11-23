import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import b3_RNN_LSTM as lstm


class LSTMPcsInitTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)

    def test_projection_is_shared_and_context_and_padding_are_preserved(self):
        weights = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        model = lstm.SequenceRateModel(
            2, 1, 2, 4, proj=3, hidden=3, cur_hidden=3, dropout=0,
            pca_weights=weights,
        ).eval()
        np.testing.assert_array_equal(model.pca_layer.weight.detach().numpy(), weights)
        self.assertTrue(torch.equal(model.pca_layer.bias, torch.zeros(2)))

        # Current features followed by two steps: feature, feature, context, valid.
        x = torch.tensor([[2., 3., 50., 70., 80., 0., 4., 5., 6., 1.]])
        seen = []
        hook = model.step_enc[0].register_forward_pre_hook(
            lambda _module, args: seen.append(args[0].detach().clone())
        )
        output = model(x)
        hook.remove()
        self.assertTrue(torch.equal(seen[0], x[:, 2:].reshape(1, 2, 4)))

        changed_padding = x.clone()
        changed_padding[:, 2:5] = torch.tensor([[-100., 200., -300.]])
        for original, changed in zip(output, model(changed_padding)):
            torch.testing.assert_close(original, changed)

        model.set_pca_frozen(True)
        self.assertFalse(model.pca_layer.weight.requires_grad)
        model.set_pca_frozen(False)
        self.assertTrue(model.pca_layer.weight.requires_grad)
        sum(rate.sum() for rate in model(x)).backward()
        self.assertIsNotNone(model.pca_layer.weight.grad)

    def test_pca_is_fit_only_on_current_features_of_fit_split(self):
        meta = {"n_current": 2, "n_series": 1, "seq_len": 1, "step_dim": 4}
        columns = ["f0", "f1", "old_f0", "old_f1", "context", "valid"]
        fit = pd.DataFrame({
            "game_id": ["20200101_A_B", "20200102_A_B", "20200103_A_B"],
            "HG": [1, 2, 1], "AG": [1, 1, 2],
            "f0": [0., 1., 2.], "f1": [2., 0., 1.],
            "old_f0": [100., 100., 100.], "old_f1": [100., 100., 100.],
            "context": [1., 1., 1.], "valid": [1., 1., 1.],
        })
        val = fit.iloc[:2].copy()
        val.loc[:, ["f0", "f1"]] = 9999.
        config = lstm.BaselineConfig(
            epochs=2, batch_size=2, seq_proj=3, seq_hidden=3, cur_hidden=3,
            module_dropout=0, pcsinit_frozen_epochs=1,
        )
        with patch.object(lstm, "load_module_meta", return_value=meta), \
             patch.object(lstm, "pca_projection", wraps=lstm.pca_projection) as projection:
            model, _ = lstm.fit_module_lstm(
                "sample", fit, val, columns, config, torch.device("cpu"), verbose=False
            )
        np.testing.assert_array_equal(projection.call_args.args[0], fit[["f0", "f1"]].to_numpy())
        self.assertIsNotNone(model.pca_layer)
        self.assertTrue(model.pca_layer.weight.requires_grad)

        config.use_pcsinit = False
        with patch.object(lstm, "load_module_meta", return_value=meta), \
             patch.object(lstm, "pca_projection") as projection:
            model, _ = lstm.fit_module_lstm(
                "sample", fit, val, columns, config, torch.device("cpu"), verbose=False
            )
        projection.assert_not_called()
        self.assertIsNone(model.pca_layer)
        self.assertEqual(model.step_enc[0].in_features, meta["step_dim"])

    def test_zero_variance_falls_back_to_unprojected_features(self):
        self.assertIsNone(lstm.pca_projection(np.ones((3, 2))))
        model = lstm.SequenceRateModel(2, 1, 1, 4, pca_weights=None)
        rates = model(torch.tensor([[1., 1., 1., 1., 0., 1.]]))
        self.assertTrue(all(torch.isfinite(rate).all() for rate in rates))


if __name__ == "__main__":
    unittest.main()