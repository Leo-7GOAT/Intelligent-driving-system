"""Query teacher on pure-learner visits; teacher never executes or trains a policy."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from navstack.config import EnvConfig
from navstack.experiments import load_run, write_json, jsonable
from navstack.imitation import collect_recovery_episode
from navstack.research import resources_ok


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    parent, output = Path(args.parent).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg_dict = json.loads((parent/'protocol.json').read_text(encoding='utf-8'))['env']
    checkpoint = parent/'supervised_checkpoint'
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT/'navstack').glob('*.py')}
    cases = [{'scenario': 'static', 'split': split, 'seed': seed}
             for split, seed in [('train', 8101), ('train', 8107), ('validation', 7301), ('validation', 7302)]]
    started = time.time()
    write_json(output/'protocol.json', {'scope': __doc__, 'cases': cases, 'model_sha256': sha(checkpoint/'model.zip'),
               'source_hashes': sources, 'script_sha256': sha(Path(__file__)), 'env': cfg_dict,
               'created_epoch': started, 'pid': os.getpid(), 'wall_seconds': 900, 'cpu_workers': 1,
               'resource_limits': {'disk_min_GiB': 12, 'ram_min_GiB': 4, 'automatic_retries': 0},
               'teacher_probability': 0., 'teacher_mode': 'current_state',
               'selection': 'Two fixed original demonstration layouts and both existing validation static cases; no successful-case selection'})
    shutil.copy2(__file__, output/'executed_script.py')
    torch.set_num_threads(1)
    model, norm = load_run(checkpoint)
    count = norm.obs_rms.count
    rows = []
    try:
        for case_id, case in enumerate(cases):
            cfg = EnvConfig.from_dict(cfg_dict)
            cfg.scenario, cfg.split = case['scenario'], case['split']
            errors = []
            with gzip.open(output/f'case_{case_id:03d}.jsonl.gz', 'wt', encoding='utf-8') as stream:
                def record(row):
                    assert row['executed_by'] == 'learner'
                    np.testing.assert_array_equal(row['executed_action'], row['learner_action'])
                    if row['teacher_query']['control'] % 20 == 0:
                        ok, reason = resources_ok(output)
                        if not ok or shutil.disk_usage(output).free < 12*1024**3 or time.time()-started > 900:
                            raise RuntimeError('Bounded learner audit: '+str(reason))
                        stream.flush()
                        write_json(output/'heartbeat.json', {'pid': os.getpid(), 'epoch': time.time(),
                                   'case_id': case_id, 'control': row['teacher_query']['control']})
                    errors.append(row['learner_action']-row['teacher_action'])
                    stream.write(json.dumps(row, default=jsonable)+'\n')
                sample = collect_recovery_episode(cfg, model, norm, case['seed'], 0., 1,
                                                 callback=record, teacher_mode='current_state')
            error = np.asarray(errors)
            rows.append({'case_id': case_id, **case, 'metrics': sample['metrics'], 'controls': len(error),
                         'teacher_queries': sample['teacher_queries'], 'teacher_query_seconds': sample['teacher_query_seconds'],
                         'mse_longitudinal_steering': (error**2).mean(0),
                         'absolute_error_p95': np.quantile(np.abs(error), .95, axis=0)})
            np.savez_compressed(output/f'case_{case_id:03d}_examples.npz', observations=sample['observations'],
                                actions=sample['actions'], returns=sample['returns'])
            write_json(output/'progress.json', rows)
            print(json.dumps(rows[-1], default=jsonable), flush=True)
        assert norm.obs_rms.count == count
        for name, digest in sources.items():
            assert sha(ROOT/name) == digest
        write_json(output/'completed.json', {'scope': __doc__, 'episodes': rows,
                   'actual_learner_controls': sum(r['controls'] for r in rows),
                   'teacher_queries': sum(r['teacher_queries'] for r in rows),
                   'teacher_executed_controls': 0, 'ppo_updates': 0, 'supervised_updates': 0})
    except BaseException as exc:
        write_json(output/'failure.json', {'epoch': time.time(), 'error': repr(exc)})
        raise
    finally:
        norm.close()


if __name__ == '__main__':
    main()
