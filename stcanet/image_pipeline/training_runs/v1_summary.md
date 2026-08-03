# STCANet + FastReID Training Run v1 — Summary

**Date:** 2026-08-03
**Config:** `configs/STCANet/Market1501_stcanet_full.yml`
**Dataset:** Market1501 (STCANetMarket1501, SCHP masks)

## Training Setup
- Backbone: build_stcanet_resnet_backbone (ResNet50 + IBN + IAT after layer2/layer3)
- Meta-arch: STCANetBaseline (CE + Triplet + Mask supervision)
- Pooling: GeneralizedMeanPooling, WITH_BNNECK: True
- Mask loss: alpha=0.5, mode='ce', warmup_epochs=10
- AMP: enabled
- Epochs: 100, batch size: 64
- Total training time: 2:41:50 (18,698 iterations)

## Evaluation Progress (no re-rank, no flip during training eval)
| Epoch | Rank-1 | mAP   |
|-------|--------|-------|
| 10    | 82.33  | 59.15 |
| 20    | 85.51  | 65.03 |
| 100   | 92.64  | 77.26 |

## Final Loss Values (epoch 100)
- total_loss: 1.115
- loss_cls: 1.035
- loss_triplet: 0 (converged, margin satisfied)
- loss_mask: 0.078 (down from 0.32 at warmup end)

## Known Issue (fixed for future runs, does not affect this run's real metrics)
Validation loss (val_loss_cls, val_total_loss) was unreliable in this run
due to a label-index misalignment bug between the train and validation
STCANetCommDataset instances (each independently computed its own
pid_dict). Fixed in build_seg.py / common_seg.py (shared_label_maps) --
not yet verified with a full training run.

## Next Steps
- [ ] Re-rank evaluation (separate --eval-only run, TEST.RERANK.ENABLED=True)
- [ ] Compare against original STCANet paper's reported mAP/Rank-1
- [ ] Compare against plain FastReID baseline (Mission 2 Step 1 sanity-check)
