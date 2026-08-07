# Consolidation — `agent/KIMODO-BRANCH-CONSOLIDATION`

Task `KIMODO-BRANCH-CONSOLIDATION-010` (2026-08-07). All history DATA-KIMODO
task branches were folded onto one coherent branch built on the
PORT-008 natural-recipe base. **Useful framework code is merged; trash
configs / task-specific generation inputs are excluded.**

- Base: `8f59b6f` (`agent/DATA-KIMODO-FRAMEWORK-PORT-008`)
- Branch: `agent/KIMODO-BRANCH-CONSOLIDATION`
- Worktree: `/home/user/CodeSpace/Diffusion/kimodo-worktrees/KIMODO-BRANCH-CONSOLIDATION`
  (the manifest's `/data/masteryip/kimodo/kimodo-agent-worktrees/...` path does
  not exist on this host; the established convention
  `/home/user/CodeSpace/Diffusion/kimodo-worktrees/` was used instead).
- Branch tip: `6513863` + the CONSOLIDATION.md commit (see git log).

## Source commits folded in (13)

| Commit | Subject | What was taken |
|---|---|---|
| `db26864` | stand_turn motion type + robot descriptions/durations | `g1_normal_loco.yaml` stand_turn motion type + robot-tuned descriptions/durations; `orchestrator.py` qpos ndarray safety |
| `a314017` | MotionSpec defaults + optional torso + YAML serialization | optional `torso_height_range`/`torso_height` (None = unconstrained), dataclass defaults, serializers optional-torso output |
| `4b6c9c4` | conservative pilot quality gate | `quality_validator.py`, `dataset_audit.py`, `g1_quality_thresholds_v1.yaml`, tests |
| `765afee` | align export names with manifest indices | `_assign_global_indices` + `sample_idx=getattr(...,"_global_idx",i)` in `orchestrator.py`, test |
| `4c1d884` | preserve global index through export | removal of the per-type local `_global_idx` overwrite (completes `765afee`) |
| `1634209` | enforce native model timing | `_validate_generation_timing` (fps/horizon guard), per-batch `seed_everything`, `analyze_quality.py`, `tail_horizon_audit.py`, `g1_quality_thresholds_v2_fps30.yaml`, tests, `PYTHONPATH=scripts` usage |
| `d6398ab` | select local NF4 encoder explicitly | `TEXT_ENCODER_MODE=local` in `scripts/env.sh` |
| `1409a8b` | label in-horizon tail plots accurately | `tail_horizon_audit.py` label fix |
| `84fa4c5` | reproducible natural-motion YAML sweep — framework parts only | `global.root2d_constraint` (enabled/stride), `build_constraints_json(enabled=...)`, planar `_speed_hint` (uses `|v|`, not `vx` alone) |
| `f5588f7` | prompt controls + evaluation | `MotionSpec.arm_swing` + `prompt_override`, `resolve_arm_swing`/exact-prompt composition, `evaluation/` package (core + cli), tests |
| `4e08430` | center-wrist-swing PCA fix | mean-centered wrist PCA in `evaluation/core.py` |
| `1d00b0d` | polar velocity ranges | `PolarVelRange`, `polar_to_cartesian`, polar `_spec_param_ranges`/LHS/uniform paths, speed/direction columns in metadata+manifest, `tests/test_polar_velocity.py` |

`3938017` (root-FS temp/cache) and `97ad86c` (gate_check heading tolerance) are
listed in the manifest but were **already ancestors of the base `8f59b6f`**
(`git merge-base --is-ancestor` = yes), so no cherry-pick was needed; both are
present on the consolidated branch through the base.

## Trash EXCLUDED (and why)

| Trash | Why excluded |
|---|---|
| `75d27b4` threshold-crossing tactical probe, `4ca964d` torso-0.70 probe, `00d8eed` matched lateral prompt sweep, `a4923bd` normal directional-arm pilot | Throwaway test-probe YAMLs (14 y6-y18 configs). |
| `g1_polar_direction_speed_24.yaml` | Task-specific polar experiment generation input; stays on the POLAR-006 task branch. |
| `configs/DATA-KIMODO-LOCOMOTION-YAML-004/y0-y5*.yaml` (14 Y-series sweep configs) | Task-specific generation inputs for LOCOMOTION-YAML-004; stay on that task branch. |
| `y10/y11_exactarms_*_tactical_*.yaml` | Tactical probe configs from f5588f7. |
| `g1_quality_pilot_v1_seed20260803/04.yaml`, `g1_quality_replacement_micro_v2_seed20260803/04.yaml` | Task-specific seed run inputs for DATA-KIMODO-001; the seed configs remain on the 001 branch. |
| `validate_DATA_KIMODO_LOCOMOTION_YAML_004.py` | Task-specific validator asserting that sweep's exact 12-config structure; dead weight without the Y-series configs. |
| `dc5eb90` "snapshot canonical dirty framework state" | Entirely superseded: identical content to `a314017` except a task-specific `g1_walk.yaml` pilot tuning (narrowed vel, 40 samples) that is NOT a framework default. Skipped. |
| DISTRIBUTED-007 `constraints/*.json` + generation assets (`2acbdd7`), 40-row `motion_types` in `g1_eight_direction_port_008.yaml`, `docs/KimodoLocoMoGen*` logos + editor dark-mode churn (`2bfba72`/`b81d216`) | Task-specific envelope/asset data and editor cosmetic churn — not framework code. |

`g1_quality_thresholds_v1.yaml` and `g1_quality_thresholds_v2_fps30.yaml` were
**kept**: they are the default threshold configs consumed by the
`quality_validator.py` framework tool and referenced by `tests/test_quality_validator.py`
(fps 50 and fps 30 variants).

## Cherry-pick -n file-level decisions

Commits that mixed useful code with trash configs were cherry-picked with `-n`,
then staged selectively:

- `4b6c9c4`: staged `quality_validator.py`, `dataset_audit.py`,
  `g1_quality_thresholds_v1.yaml`, `tests/test_quality_validator.py`.
  Dropped `g1_quality_pilot_v1_seed*.yaml`.
- `1634209`: staged README, `analyze_quality.py`, `tail_horizon_audit.py`,
  `g1_quality_thresholds_v2_fps30.yaml`, `orchestrator.py`,
  `tests/test_quality_validator.py`. Dropped
  `g1_quality_replacement_micro_v2_seed*.yaml`.
- `84fa4c5`: staged README, `config.py`, `constraints.py`, `orchestrator.py`,
  `prompts.py`. Dropped the 14 `y*.yaml` sweep configs and
  `validate_DATA_KIMODO_LOCOMOTION_YAML_004.py`.
- `f5588f7`: staged README, `config.py`, `prompts.py`, `sampler.py`,
  `evaluation/*`, both tests. Dropped `y10/y11_exactarms_*_tactical_*.yaml`.
- `1d00b0d`: staged README, `config.py`, `editor/serializers.py`,
  `orchestrator.py`, `sampler.py`, `tests/test_polar_velocity.py`. Dropped
  `g1_polar_direction_speed_24.yaml`.
- `db26864`: full commit kept (both files are framework config/code).
- `dc5eb90`: not cherry-picked (superseded, see trash table).

## Conflict resolutions

- `a314017` × PORT-008 in `sampler.py`: kept PORT-008's `_prompt_for_spec` /
  `_extra_kwargs` (natural-recipe prompt override + Root2D kwargs) alongside
  a314017's optional `torso_height` (`Optional[float] = None`), guarding the
  torso handling in `_prompt_for_spec`, `_spec_param_ranges`, `_lhs_sample_from_spec`
  and `sample_params`.
- `84fa4c5` in `config.py`: kept PORT-008 `cfg_type`/`cfg_weight` alongside the
  new `global.root2d_constraint`. In `constraints.py`: merged the natural-recipe
  `stride` docstring with the new `enabled` parameter.
- `f5588f7` in `config.py`/`prompts.py`/`sampler.py`: kept PORT-008 `prompt`
  (verbatim override) AND polar `prompt_override`/`arm_swing`. `load_config`
  sets both from the same YAML `prompt` key; `build_motion_prompt` gained
  `speed_hint` + `exact_prompt` + `arm_swing`; `_prompt_for_spec` routes
  `prompt_override`/`arm_swing` through to `build_motion_prompt`.
- `d6398ab` in `scripts/env.sh`: merged the base's root-FS temp/cache echo line
  with the new `TEXT_ENCODER_MODE` echo (one combined line).
- `765afee`/`4c1d884` × PORT-008 `orchestrator.py`: `4c1d884`'s deletion applied
  because `db26864` had added the `_global_idx = i` line earlier in the chain;
  net result is global indices assigned once via `_assign_global_indices` and
  never overwritten locally.
- `1d00b0d`: auto-merged onto the PORT-008+natural+MotionSpec-defaults state
  with **no conflicts**; Cartesian `vel_cmd` and `polar_vel_cmd` are
  mutually-exclusive alternatives (validated in `_parse_vel_cmd`).

## Backward-compat gate results (2026-08-07, host, no GPU)

- Configs load + sample (config→MotionSampler, LHS + uniform where applicable):
  `debug_squat_height.yaml` (5 types/100), `g1_normal_loco.yaml` (5/250),
  `g1_walk.yaml` (1/200), `g1_eight_direction_port_008.yaml` (40/40),
  `g1_polar_direction_speed_24.yaml` (8/24 — config kept only for the gate
  test; not committed to the branch, per the exclusion list).
- Tests: `tests/test_polar_velocity.py` (5), `tests/locomotion_framework/`
  (4), `tests/test_quality_validator.py` (9) → **18 passed**.
- `scripts/locomotion_framework/run_distributed.sh`: `bash -n` OK; its embedded
  Python only needs `yaml` + `g1_normal_loco.yaml` (present).
- LocoEditor: all framework modules `py_compile` clean; `editor/serializers.py`
  round-trips natural + polar configs (`config_to_yaml_str` ↔
  `yaml_str_to_config`). The editor's `torch`/viser GPU deps are environment
  dependencies, unchanged from the base.
- Completion fix: polar source branch never serialized `polar_vel_cmd` back to
  YAML (round-trip data loss); added polar output to `config_to_yaml_str`
  (commit `6513863`).

## Archive tags (source branches folded in)

Annotated archive tags were created on the source-branch tips and pushed to
`fork` (2026-08-07). They record the reason so nobody mistakes these branches
for active work. Branches are NOT deleted.

| Tag | Target | Reason summary |
|---|---|---|
| `archive/agent-20260806-data-kimodo-polar-distribution-006@20260807` | `1d00b0d` | polar/wrist-PCA/prompt-controls/YAML-sweep-framework/MotionSpec-defaults folded in; `g1_polar_direction_speed_24.yaml` + Y-series/probes excluded |
| `archive/agent-data-kimodo-distributed-007@20260807` | `97ad86c` | root-FS temp/cache + gate_check already in base `8f59b6f`; DISTRIBUTED-007 constraints/assets excluded |
| `archive/agent-data-kimodo-locomotion-yaml-004@20260807` | `00d8eed` | stand_turn / YAML-sweep-framework / prompt-controls / wrist-PCA folded in; y0-y18 + tactical probes excluded |
| `archive/agent-data-kimodo-textonly-002@20260807` | `a314017` | MotionSpec defaults folded in |
| `archive/agent-data-kimodo-constraint-003@20260807` | `a314017` | MotionSpec defaults folded in |
| `archive/agent-data-kimodo-001-constraint-ablation@20260807` | `1409a8b` | quality gate / export index+name / native timing / NF4 / metrics folded in; `*_seed*.yaml` excluded |
| `archive/agent-data-kimodo-001-quality-gen@20260807` | `1409a8b` | same as the 001-constraint-ablation tag |
| `archive/agent-data-kimodo-framework-port-008@20260807` | `8f59b6f` | BASE of the consolidation |

Not tagged archived: `agent/DATA-KIMODO-VELOCITY-DISTRIBUTION-009` (active,
running agent, untouched); `feat/mogen_distributed` (local branch at `2bfba72`,
an ancestor already in the base — its polar work lives on the fork POLAR-006
branch, which is tagged).

## Notes / coordination

- `agent/DATA-KIMODO-VELOCITY-DISTRIBUTION-009` was NOT touched. PM will merge
  it on top of this branch; its `config.py`/`sampler.py`/`orchestrator.py`
  win in any overlap. This branch intentionally kept the PORT-008 natural
  recipe + polar features reconciled so the merge is clean.
