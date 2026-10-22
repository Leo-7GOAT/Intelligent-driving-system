# Research results and claim boundaries

**Research acceptance remains unmet.** The [acceptance ledger](research_acceptance.md) retains the original targets; a development pre-release does not lower them.

Detailed current cases are in [the development report](../reports/development_20261022.md). Historical H/B1 lightweight results comprise three complete1M pairs: H30/40 andB1 0/40 for each. B3 also achieves30/40; two H seeds are incomplete. Historical CARLA adaptation has two paired seeds at approximately50k world steps, H9/30 and10/30 versusB1 0/30, with H successes confined to dynamic cases. These results do not establish a healthy-baseline hierarchy advantage or PPO necessity.

The new single-seed feedback development has H/H0 completion6/10 each at100k,8/10 versus6/10 at200k, and8/10 each at300k. Both frozen200k policies score2/6 in actual CARLA transfer with calibrated vehicle geometry. Current longer lightweight runs and CARLA adaptation are separate, explicitly labeled development batches.

No formal paired95% confidence interval is reported for this single-seed development. Repeated conditions from one model are not independent training seeds. Completion, interventions and progress are reported jointly; failed episodes stay in the denominator.

The release asset reproduces a declared200k development checkpoint rather than a best-seed or best-test selection. Its complete20-episode CPU replay was verified locally against the recorded outcomes. A fresh GitHub clone independently installed pinned dependencies, passed66CPU tests with5conditional skips, and reproduced all20 cases. Release SHA and asset download status are recorded separately in the release record.
