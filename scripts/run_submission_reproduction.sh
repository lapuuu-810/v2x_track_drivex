#!/usr/bin/env bash
# =============================================================================
# v17 (Combined 0.7421, submission #886382) one-command reproduction.
#   realism (S12-x092) -> six-model inference -> v17 fusion -> package -> verify
# All paths are relative to the project root. Derived data is intentionally not
# included; generate it with docs/REPRODUCTION_GUIDE.md or provide prepared data.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
ENV_NAME="${CONDA_ENV:-eccv-v2x-submission}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/repro_v17}"
DEVICE="${DEVICE:-0}"
SKIP_INFER="${SKIP_INFER:-0}"
SKIP_REALISM="${SKIP_REALISM:-0}"
PARALLEL="${PARALLEL:-0}"

TEST_FRAMES="$ROOT/competition/starting_kit/detection_test_frames"
FRAME_IDS="$ROOT/competition/starting_kit/detection_test_frame_ids.json"
BASE_SYNTHETIC="$ROOT/outputs/current_submission_v2_synthetic/synthetic"
REF_PREDS="$ROOT/outputs/final_fusion/predictions_final_v17.json"   # optional reference
EXPECTED_FRAME1_SHA="3446f74dcbcb9caa330f208768d8ce362fa76d288d1ad7f745d27f1c11e36b40"
EXPECTED_BOXES=4177
EXPECTED_CLASS="vehicle:1157 pedestrian:2320 truck:700"

WEIGHTS_DIR="$ROOT/weights"
CK_B4="$WEIGHTS_DIR/dsvt_b4_epoch60.pth"
CK_B7="$WEIGHTS_DIR/dsvt_b7_epoch60.pth"
CK_B10="$WEIGHTS_DIR/dsvt_b10_epoch60.pth"
CK_B11="$WEIGHTS_DIR/dsvt_b11_epoch60.pth"
CK_D14="$WEIGHTS_DIR/dsvt_d14_epoch40.pth"
CK_D50="$WEIGHTS_DIR/dsvt_d50_epoch40.pth"

CFG_B4="$ROOT/configs/dsvt_pedalign.yaml"
CFG_B7="$ROOT/configs/dsvt_pedalign_corner.yaml"
CFG_B10="$ROOT/configs/dsvt_pedalign_corner_015.yaml"
CFG_B11="$ROOT/configs/dsvt_pedalign_center_weighted.yaml"
CFG_D14="$ROOT/configs/dsvt_rendered_base.yaml"
CFG_D50="$ROOT/configs/dsvt_rendered_dense.yaml"

S12_DIR="$OUT_DIR/synthetic_s12x092"
DECL="$OUT_DIR/declaration.txt"
PY_RUN="conda run -n $ENV_NAME --no-capture-output python"

mkdir -p "$OUT_DIR"
cd "$ROOT"

echo "=============================================================="
echo " v17 reproduction (self-contained) | root=$ROOT | device=$DEVICE"
echo "=============================================================="

# ---- 0) sanity ---------------------------------------------------------------
for f in "$CK_B4" "$CK_B7" "$CK_B10" "$CK_B11" "$CK_D14" "$CK_D50"; do
    [ -f "$f" ] || { echo "MISSING checkpoint: $f"; exit 1; }
done
[ -d "$TEST_FRAMES" ] || { echo "MISSING test frames: $TEST_FRAMES"; exit 1; }

# ---- 1) realism: S12-x092 ------------------------------------------------------
if [ "$SKIP_REALISM" = "1" ] && [ -f "$S12_DIR/0001.bin" ]; then
    echo "[1/4] realism: reuse existing $S12_DIR"
else
    if [ ! -f "$BASE_SYNTHETIC/0001.bin" ]; then
        echo "ERROR: realism base frames not found: $BASE_SYNTHETIC"
        echo "  Prepared data is intentionally NOT shipped. Two options:"
        echo "   a) run the reproduction guide (docs/REPRODUCTION_GUIDE.md) to regenerate it,"
        echo "   b) place your prepared 50 base frames under outputs/current_submission_v2_synthetic/synthetic"
        exit 1
    fi
    echo "[1/4] realism: generating S12-x092 (affine x0.92/y1.0/z0.85)"
    $PY_RUN src/transform_anchor_points.py \
        --input "$BASE_SYNTHETIC" --output "$S12_DIR" \
        --x-scale 0.92 --y-scale 1.00 --z-scale 0.85
fi
sha1=$(sha256sum "$S12_DIR/0001.bin" | cut -d' ' -f1)
echo "      frame1 sha256: $sha1"
[ "$sha1" = "$EXPECTED_FRAME1_SHA" ] || { echo "ERROR: S12-x092 frame 1 hash mismatch"; exit 1; }
echo "      OK: realism frames identical to submitted (frame1 hash)"

# ---- 2) inference ---------------------------------------------------------------
if [ "$SKIP_INFER" = "1" ]; then
    echo "[2/4] inference: SKIPPED (requires per-model test predictions in $OUT_DIR)"
    for n in preds_b4_test.json preds_b7_test.json preds_b10_test.json preds_b11_test.json preds_d14_test.json preds_d50_test.json; do
        [ -f "$OUT_DIR/$n" ] || { echo "ERROR: $OUT_DIR/$n missing (not shipped; run inference or place your own)"; exit 1; }
    done
else
    run_infer() {
        local name="$1" cfg="$2" ckpt="$3" out="$4"
        echo "[2/4] infer $name -> $out"
        $PY_RUN src/final_submission_prepare.py infer-test \
            --config "$cfg" --checkpoint "$ckpt" \
            --frames "$TEST_FRAMES" --frame-ids "$FRAME_IDS" \
            --output "$out" --device "$DEVICE"
    }
    if [ "$PARALLEL" = "1" ]; then
        echo "WARNING: PARALLEL unsupported (infer-test only supports device 0); running sequentially."
    fi
    run_infer b4  "$CFG_B4"  "$CK_B4"  "$OUT_DIR/preds_b4_test.json"
    run_infer b7  "$CFG_B7"  "$CK_B7"  "$OUT_DIR/preds_b7_test.json"
    run_infer b10 "$CFG_B10" "$CK_B10" "$OUT_DIR/preds_b10_test.json"
    run_infer b11 "$CFG_B11" "$CK_B11" "$OUT_DIR/preds_b11_test.json"
    run_infer d14 "$CFG_D14" "$CK_D14" "$OUT_DIR/preds_d14_test.json"
    run_infer d50 "$CFG_D50" "$CK_D50" "$OUT_DIR/preds_d50_test.json"
fi

# ---- 3) fusion + packaging --------------------------------------------------------
echo "[3/4] assembling v17 fusion (veh ens 0.5 / trk 0.2 / ped B4 + blends 0.2/0.2/0.25 + final NMS 0.2@0.03)"
cat > "$DECL" <<EOF
I confirm that I did not train on any frame listed in forbidden_frames.txt,
and that all submitted detections were produced by a model trained only on
synthetic data I generated.

Team / individual name: ${SUBMISSION_AUTHOR:-submission team}
Date (YYYY-MM-DD):       ${SUBMISSION_DATE:-2026-08-07}
Submission tag:          ${SUBMISSION_TAG:-DSVT-PedAlignV2-TTA4-final-v4}
EOF
$PY_RUN "$SCRIPT_DIR/assemble_submission.py" \
    --test-frames "$TEST_FRAMES" --frame-ids "$FRAME_IDS" \
    --b4 "$OUT_DIR/preds_b4_test.json" --b7 "$OUT_DIR/preds_b7_test.json" \
    --b10 "$OUT_DIR/preds_b10_test.json" --b11 "$OUT_DIR/preds_b11_test.json" \
    --d14 "$OUT_DIR/preds_d14_test.json" --d50 "$OUT_DIR/preds_d50_test.json" \
    --synthetic "$S12_DIR" --declaration "$DECL" \
    --output-dir "$OUT_DIR"

# ---- 4) verification -------------------------------------------------------------
echo "[4/4] verification"
python3 - "$OUT_DIR" "$REF_PREDS" <<'EOF'
import json, sys, collections
out_dir, ref_path = sys.argv[1], sys.argv[2]
new = json.load(open(f"{out_dir}/predictions_final_v17.json"))
c = collections.Counter()
for v in new.values():
    for b in v:
        c[b["class"]] += 1
total = sum(c.values())
ok = total == 4177 and c["vehicle"] == 1157 and c["pedestrian"] == 2320 and c["truck"] == 700
print(f"boxes={total} per_class={{vehicle:{c['vehicle']}, pedestrian:{c['pedestrian']}, truck:{c['truck']}}}")
print("counts check:", "PASS (4177 / 1157 / 2320 / 700)" if ok else "FAIL")
if ok and ref_path and __import__("os").path.exists(ref_path):
    ref = json.load(open(ref_path))
    n_box = n_exact = n_close = n_diff = 0
    max_dev = 0.0
    for fid in ref:
        a, b = new.get(fid, []), ref[fid]
        if len(a) != len(b):
            print(f"FRAME {fid}: box count {len(a)} vs {len(b)}"); n_diff += 1; continue
        for ra, rb in zip(a, b):
            n_box += 1
            if ra == rb:
                n_exact += 1
            else:
                d = max(abs(x - y) for x, y in zip(ra["box"], rb["box"]))
                ds = abs(ra["score"] - rb["score"])
                max_dev = max(max_dev, d, ds)
                if ra["class"] == rb["class"] and ds < 1e-4 and d < 1e-3:
                    n_close += 1
                else:
                    n_diff += 1
                    if n_diff <= 3: print("DIFF", fid, ra, rb)
    print(f"reference compare: boxes={n_box} exact={n_exact} close={n_close} different={n_diff} max_dev={max_dev:.2e}")
    print("reference compare:", "PASS (1:1 correspondence, float noise only)" if n_diff == 0 else "FAIL")
if not ok:
    sys.exit(1)
EOF
echo "=============================================================="
echo " DONE. artifacts in $OUT_DIR"
ls -la "$OUT_DIR"
echo "=============================================================="
