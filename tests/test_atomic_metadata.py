import json
from concurrent.futures import ThreadPoolExecutor
from navstack.research import atomic_json


def test_concurrent_status_writers_never_share_temporary_file(tmp_path):
    target=tmp_path/"status.json"
    def writer(rank):
        for step in range(30):atomic_json(target,{"rank":rank,"step":step,"payload":[rank]*20})
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(writer,range(4)))
    data=json.loads(target.read_text(encoding="utf-8"))
    assert data["payload"]==[data["rank"]]*20 and data["step"]==29
    assert not list(tmp_path.glob("*.tmp"))
