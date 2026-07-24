#!/usr/bin/env bash
# setup_laptop_env.sh — one-shot environment setup for xr_teleoperate on the expo laptop.
# Assumes the expo zip is already extracted so these exist under $HOME:
#   ~/xr_teleoperate  ~/unitree_sdk2_python  ~/cyclonedds/install
# Safe to re-run. Does NOT configure networking (see NETWORK_SETUP.md / expo plan Part B/C).
set -euo pipefail

ENV_NAME="tv-2"
REPO="$HOME/xr_teleoperate"
SDK="$HOME/unitree_sdk2_python"
CDDS="$HOME/cyclonedds/install"
YML="$REPO/tv-2_portable.expo.yml"
REQ="$REPO/tv-2-requirements.expo.txt"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# 1. Prerequisites -----------------------------------------------------------
say "Checking prerequisites"
command -v conda >/dev/null 2>&1 || die "conda not found. Install Miniconda/Anaconda first."
for d in "$REPO" "$SDK" "$CDDS"; do
  [ -d "$d" ] || die "Missing required folder: $d  (did the zip extract to \$HOME?)"
done
[ -f "$YML" ] || die "Missing env spec: $YML"
ls "$CDDS"/lib/libddsc.so* >/dev/null 2>&1 || die "libddsc.so not found under $CDDS/lib"

# 2. CycloneDDS env vars (idempotent append to ~/.bashrc + export for THIS run) ----
say "Configuring CycloneDDS env vars in ~/.bashrc"
add_line() { grep -qxF "$1" "$HOME/.bashrc" || echo "$1" >> "$HOME/.bashrc"; }
add_line 'export CYCLONEDDS_HOME=$HOME/cyclonedds/install'
add_line 'export LD_LIBRARY_PATH=$HOME/cyclonedds/install/lib:$LD_LIBRARY_PATH'
export CYCLONEDDS_HOME="$CDDS"
export LD_LIBRARY_PATH="$CDDS/lib:${LD_LIBRARY_PATH:-}"

# 3. Make `conda activate` usable inside this non-interactive script ----------
say "Initializing conda shell hook"
source "$(conda info --base)/etc/profile.d/conda.sh"

# 4. Create the conda env (skip if it already exists) ------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  say "Conda env '$ENV_NAME' already exists — skipping create (delete with 'conda env remove -n $ENV_NAME' to rebuild)"
else
  say "Creating conda env '$ENV_NAME' from $(basename "$YML")"
  conda env create -n "$ENV_NAME" -f "$YML" \
    || die "conda env create failed. Fallback: create an empty env (conda create -n $ENV_NAME python=3.10),
             then 'pip install -r $REQ', then re-run this script."
fi
conda activate "$ENV_NAME"

# 5. Re-point the editable local packages at the laptop's paths --------------
# (Not on PyPI; the yml/requirements deliberately exclude them.)
say "Installing editable local packages"
pip install -e "$SDK"
pip install -e "$REPO/teleop/televuer"
pip install -e "$REPO/teleop/teleimager"
pip install -e "$REPO/teleop/robot_control/dex-retargeting"
# Inspire hands only (expo uses BrainCo, so normally skip):
# [ -d "$HOME/inspire_hand_ws" ] && pip install -e "$HOME/inspire_hand_ws/inspire_hand_sdk"

# 6. Verify — native DDS lib loads AND every teleop import resolves -----------
say "Verifying installation"
python - <<'PY'
from cyclonedds import core                                    # loads libddsc.so via CYCLONEDDS_HOME
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import televuer, teleimager
from teleop.robot_control.robot_arm_ik import G1_23_ArmIK
print("cyclonedds native OK; unitree_sdk2py/televuer/teleimager/IK imports OK")
PY

say "DONE. Env '$ENV_NAME' is ready."
echo "Next: open a NEW terminal (to pick up ~/.bashrc), then 'conda activate $ENV_NAME'"
echo "and continue with Part B (Ethernet) and Part C (phone hotspot)."
