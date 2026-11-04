# Public native PPO RGB evidence

[Development prerelease](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/tag/v0.2.0-dev.20261101),
source `ba932787113168ad5781ee640c6f06d09bbee0e1`.
The [executed receipt](../reports/native_PPO_RGB_public_verification_20261101.json)
contains the public manifests, archive/video SHA256s and independent results.
All four videos include their method, seed, batch and original case in the name.

H succeeds on both selected known roads; B1 succeeds on one and leaves bounds on
the other. These are frozen, previously PPO-trained seed11 policies, with zero
updates during recording. The two known cases are development illustrations.
Palm canopy partly occludes the second road. Original PNG pixels remain local;
their hashes/timestamps, encoded videos, complete control traces, models,
normalizers, source, configurations and companion figures are published.

From a fresh checkout, install the pinned CPU dependencies as described in the
main README, then run this PowerShell block at the repository root. It downloads
both public capsules without GitHub login, verifies their hashes, reconstructs
all raw/model/figure results and decodes every video frame. Outputs use a fresh
directory. Verification creates no CARLA interactions or optimizer updates.

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-video.txt
if ($LASTEXITCODE) { throw 'Video dependency installation failed' }
$rgbReceipt = Get-Content -Raw reports/native_PPO_RGB_public_verification_20261101.json | ConvertFrom-Json
$rgbOutput = 'runs/public_RGB_' + (Get-Date -Format yyyyMMdd_HHmmss_fff)
foreach ($rgbCapsule in $rgbReceipt.capsules) {
    $rgbFolder = "$rgbOutput/$($rgbCapsule.method)"
    .venv/Scripts/python.exe scripts/download_native_asset.py --manifest-url $rgbCapsule.public_manifest_url --sha256 $rgbCapsule.archive.sha256 --output "$rgbFolder/download"
    if ($LASTEXITCODE) { throw 'Public capsule download failed' }
    .venv/Scripts/python.exe scripts/reproduce_native_asset.py --archive "$rgbFolder/download/$($rgbCapsule.archive.archive)" --sha256 $rgbCapsule.archive.sha256 --output "$rgbFolder/rebuilt"
    if ($LASTEXITCODE) { throw 'Raw/model/figure reproduction failed' }
    $rgbBatch = @(Get-ChildItem -LiteralPath "$rgbFolder/rebuilt/asset" -Directory)
    if ($rgbBatch.Count -ne 1) { throw 'Unexpected extracted batch inventory' }
    Copy-Item -LiteralPath "$($rgbBatch[0].FullName)/videos" -Destination "$rgbFolder/video_decode" -Recurse
    .venv/Scripts/python.exe scripts/verify_carla_videos.py --videos "$rgbFolder/video_decode"
    if ($LASTEXITCODE) { throw 'Video decode verification failed' }
}
```

Expected totals: H 514 files, 645 applied controls, 647 video frames; B1 506 files,
1,183 controls, 1,185 frames. Each model reproduces both measured-start actions;
both reconstructed figures are byte-identical. Video decoding uses a separate
copy to preserve the original exported verification receipts. The actual public
download and independent reconstruction already passed for both capsules.
