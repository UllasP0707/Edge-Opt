"""Small, runnable QAT + polynomial pruning example (requires edge-opt[torch])."""

from __future__ import annotations

import torch
import torch.nn as nn

from edge_opt import (
    PolynomialPruningSchedule,
    QATConfig,
    TorchMagnitudePruner,
    convert_qat,
    export_int8_bundle,
    freeze_qat_observers,
    prepare_qat,
)


class TinyEncoderDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(32, 64)
        self.activation = nn.ReLU()
        self.decoder = nn.Linear(64, 16)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.activation(self.encoder(values)))


def main() -> None:
    torch.manual_seed(7)
    model, report = prepare_qat(TinyEncoderDecoder(), QATConfig())
    print(f"Inserted fake quantization into {report.total_modules} modules")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_function = nn.MSELoss()
    pruner = TorchMagnitudePruner(
        PolynomialPruningSchedule(
            final_sparsity=0.65,
            begin_step=0,
            end_step=40,
            update_frequency=2,
        )
    )

    for step in range(41):
        inputs = torch.randn(8, 32)
        targets = torch.randn(8, 16)
        pruner.update(model, step)
        optimizer.zero_grad()
        loss = loss_function(model(inputs), targets)
        loss.backward()
        pruner.mask_gradients(model)
        optimizer.step()
        pruner.enforce(model)

    freeze_qat_observers(model)
    converted = convert_qat(model)
    manifest = export_int8_bundle(converted, "artifacts/tiny-int8")
    print(f"Exported {manifest}")


if __name__ == "__main__":
    main()

