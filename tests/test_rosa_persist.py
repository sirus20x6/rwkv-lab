import sys, types, torch, torch.nn as nn
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from pathlib import Path
import convert_train
from rosa_soft_layer import RosaAnchorLayer
from rosa_soft import RosaAnchorScaleConfig, RosaAnchorScaleController

out = Path(__import__("tempfile").mkdtemp(prefix="rosa_persist_test_"))
args = types.SimpleNamespace(prior_rwkv_layers="", model_dir="m", patch_dir="", layer=16,
                             save_optimizer=False, optimizer="spectral_muon")
student, codec = nn.Linear(4, 4), nn.Linear(4, 4)
rosa = RosaAnchorLayer(256, M=4, window_size=32, route_dim=128)
rosa.scale = 0.0421  # controller-calibrated value
ctl = RosaAnchorScaleController(RosaAnchorScaleConfig(seq_len=1024, qk_bits=4, window_size=32))
ctl.top_prob_ema = 0.7311

sd = convert_train._save_ckpt(out, 500, student, codec, args, opt=None, rosa_soft=rosa, rosa_ctl=ctl)
blob = torch.load(sd / "ckpt.pt", map_location="cpu", weights_only=False)
assert blob["rosa_soft_scale"] == 0.0421, blob.get("rosa_soft_scale")
assert blob["rosa_ctl_ema"] == 0.7311, blob.get("rosa_ctl_ema")
assert "rosa_soft" in blob
print(f"save side: rosa_soft_scale={blob['rosa_soft_scale']} rosa_ctl_ema={blob['rosa_ctl_ema']} OK")

# restore side: mirror build()'s rosa_restore + train()'s controller seeding
rosa_restore = {"scale": blob.get("rosa_soft_scale"), "ema": blob.get("rosa_ctl_ema")}
ctl2 = RosaAnchorScaleController(RosaAnchorScaleConfig(
    seq_len=1024, qk_bits=4, window_size=32,
    initial_scale=(None if None is not None else rosa_restore.get("scale"))))
ctl2.top_prob_ema = float(rosa_restore["ema"])
assert ctl2.scale == 0.0421 and ctl2.top_prob_ema == 0.7311
# vs the old behavior: fresh estimate would differ
fresh = RosaAnchorScaleController(RosaAnchorScaleConfig(seq_len=1024, qk_bits=4, window_size=32)).scale
print(f"restore side: controller resumes scale={ctl2.scale} ema={ctl2.top_prob_ema} "
      f"(old behavior would re-estimate {fresh:.4g}) OK")
# _save_best path (async writer) with the same keys
convert_train._save_best(out, 501, 9.9, student, codec, args, opt=None, rosa_soft=rosa, rosa_ctl=ctl)
convert_train._flush_best_saves()
bb = torch.load(out / "best" / "ckpt.pt", map_location="cpu", weights_only=False)
assert bb["rosa_soft_scale"] == 0.0421 and bb["rosa_ctl_ema"] == 0.7311
print("best/ ckpt carries scale+ema too OK")
