# Research results and claim boundaries

**Research acceptance remains unmet.** The [acceptance ledger](research_acceptance.md) retains the original targets; a development pre-release does not lower them.

Detailed current cases are in [the development report](../reports/development_20261022.md). Historical H/B1 lightweight results comprise three complete1M pairs: H30/40 andB1 0/40 for each. B3 also achieves30/40; two H seeds are incomplete. Historical CARLA adaptation has two paired seeds at approximately50k world steps, H9/30 and10/30 versusB1 0/30, with H successes confined to dynamic cases. These results do not establish a healthy-baseline hierarchy advantage or PPO necessity.

The v8 single-seed feedback development has H/H0 completion6/10 each at100k,8/10 versus6/10 at200k,8/10 each at300k, and8/10 versus9/10 at1M. Both frozen200k policies score2/6 in actual CARLA transfer with calibrated vehicle geometry; adapting those same200k policies for approximately100k actual world ticks each gives3/6 each. There is no established feedback advantage. Strong B3 on the same v8 lightweight development cases completes10/10, including both unseen cases failed by H. Its rule-selection compute and full traces are retained.

The new v12 experiment independently trains shared candidate-scoring H/H0 policies from scratch, keeping reward, action catalog and low level fixed. It exposes explicit target geometry with fixed scales; its matched ablation removes candidate witnesses and previous outcomes. At25k, H completes0/10 and H0 4/10; at100k, H completes8/10 and H0 6/10. Fixed100k CARLA calibrated transfer gives4/6 versus3/6; equal-budget real CARLA adaptation has started. These small developmental differences are not confirmed benefits. The v11 flat baseline is a separate450-input experiment (0/10 at100k and300k) and cannot be silently treated as the final605-input matched comparator.

No formal paired95% confidence interval is reported for this single-seed development. Repeated conditions from one model are not independent training seeds. Completion, interventions and progress are reported jointly; failed episodes stay in the denominator.

The release asset reproduces a declared200k development checkpoint rather than a best-seed or best-test selection. Its complete20-episode CPU replay was verified locally against the recorded outcomes. A fresh GitHub clone independently installed pinned dependencies, passed66CPU tests with5conditional skips, and reproduced all20 cases. Release SHA and asset download status are recorded separately in the release record.
