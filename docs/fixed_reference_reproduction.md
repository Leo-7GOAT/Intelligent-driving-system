# Reproduce the fixed-reference controller study

[Public-download independent reproduction receipt](../reports/fixed_reference_public_reproduction_20261031.json): all145file digests,38,962state transitions,metrics and PNG match at checkoutcd989d9. The initially failed line-ending check is retained in that receipt.

This is a lightweight VehiclePlant comparison, not CARLA or PPO evidence. All
80 completed episodes and the earlier failed version's 20 completed episodes
are retained. Only the complete version contributes to the reported scores.
Both controllers complete 40/40 conditions; Pursuit/PI has lower lateral error
and CPU cost, while MPC has lower jerk. No uniform MPC advantage is claimed.

Download both `.part001` and `.part002` files beside the
[public manifest](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/v0.2.0-dev.20261022/fixed_reference_controllers_20261031_repro.zip.parts.json)
into a new directory. With the repository's locked CPU environment installed,
run from its root (replace `downloads` with that directory):

```powershell
python scripts/native_asset_parts.py join --manifest downloads/fixed_reference_controllers_20261031_repro.zip.parts.json --output downloads/controllers.zip --sha256 6b087cdae7b97ecdd9360e0ef75c66c6ccae7de089f1dfd9f24ea340753c107f
python scripts/fixed_reference_asset.py reproduce --archive downloads/controllers.zip --sha256 6b087cdae7b97ecdd9360e0ef75c66c6ccae7de089f1dfd9f24ea340753c107f --output runs/controller_public_reproduction
```

The ZIP contains 145 manifest-bound files, including original registered source
bytes for both versions, all traces, references, measured road scenes,
calibration and the baseline figure. The command verifies the archive and every
file, replays all 38,962 complete-study state transitions, recomputes metrics and
checks the PNG byte-for-byte. The failed prefix's 10,190 controls are preserved
and integrity-checked, but excluded from this complete-study replay and scores.
Five initial-error seeds are not five independently trained neural policies.
No simulator, optimizer update or new physical execution is involved.
