# Changelogs <!-- omit in toc -->

## v1.1.0 (2022-09-15)

Features:

- Introduce `add_value_for_multi_episode_process()`, a multi-episodic value
  column calculation (#19). Credit to @Laurenstc for contributing.

- Remove dependency to `stable_baselines3` due to conflicting version of `gym`
  required (#18). As a consequence, `generate_data_gym()` is no longer part of
  `a2rl` library, and is instead provided as an sample fragment in the
  data-property example.

Bug fixes:

- Remove extraneous smoothing terms in `add_value()` (#21). Credit to
  @gballardin for noticing.

## v1.0.2 (2022-09-07)

- Update MDP check to empirical version

## v1.0.1 (2022-08-31)

- Updated documentations

## v1.0.0 (2022-08-31)

- Initial release
