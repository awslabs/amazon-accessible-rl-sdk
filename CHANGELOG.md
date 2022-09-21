# Changelogs <!-- omit in toc -->

## v1.1.0 (2022-09-21)

Bug fixes:

- The warning silencer from `a2rl` crashes on `matplotlib`'s `DeprecationWarning`
  ([#26](https://github.com/awslabs/amazon-accessible-rl-sdk/pull/26)).

## v1.1.0 (2022-09-15)

Features:

- Introduce `add_value_for_multi_episode_process()`, a multi-episodic value column calculation
  ([#19](https://github.com/awslabs/amazon-accessible-rl-sdk/pull/19)). Credit to
  [@Laurenstc](https://github.com/Laurenstc) for contributing.

- Remove dependency to `stable_baselines3` due to conflicting version of `gym` required
  ([#18](https://github.com/awslabs/amazon-accessible-rl-sdk/pull/18)).

  Consequently, `generate_data_gym()` is no longer part of `a2rl` library, and is instead provided
  as a sample fragment in the data-property example.

Bug fixes:

- Remove extraneous smoothing terms in `add_value()`
  ([#21](https://github.com/awslabs/amazon-accessible-rl-sdk/pull/21)). Credit to
  [@gballardin](https://github.com/gballardin) for reporting.

## v1.0.2 (2022-09-07)

- Update MDP check to empirical version

## v1.0.1 (2022-08-31)

- Updated documentations

## v1.0.0 (2022-08-31)

- Initial release
