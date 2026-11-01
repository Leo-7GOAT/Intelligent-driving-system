from copy import deepcopy
import pytest
from scripts.publish_native_matched_feedback_cohort import require_rows,missing_objects,decode_command_output


def rows():return [{'seed':s,'batch':'pair'+str(s),'protocol_sha256':'bound'} for s in [11,22,33,44,55]]


@pytest.mark.parametrize('change',['missing','duplicate','reordered','unsigned'])
def test_publication_never_omits_or_replaces_a_registered_seed(change):
    data=rows()
    if change=='missing':data.pop()
    if change=='duplicate':data[-1]['batch']=data[0]['batch']
    if change=='reordered':data.reverse()
    if change=='unsigned':data[2].pop('protocol_sha256')
    with pytest.raises(ValueError):require_rows(data)


def test_only_absent_objects_are_uploaded_and_matching_objects_are_reused():
    items=[{'name':'part001','bytes':10,'sha256':'abc'},{'name':'part002','bytes':5,'sha256':'def'}]
    assets={'part001':{'state':'uploaded','size':10,'digest':'sha256:abc'}}
    assert missing_objects(items,assets)==[items[1]]
    assert require_rows(rows())==rows()


@pytest.mark.parametrize('change',['size','digest','state'])
def test_foreign_or_partial_remote_object_is_never_treated_as_replaceable(change):
    item={'name':'part001','bytes':10,'sha256':'abc'}
    remote={'state':'uploaded','size':10,'digest':'sha256:abc'}
    remote[change]={'size':9,'digest':'sha256:other','state':'starter'}[change]
    before=deepcopy(remote)
    with pytest.raises(ValueError):missing_objects([item],{'part001':remote})
    assert remote==before


def test_windows_ci_checkmarks_use_utf8_without_losing_original_bytes():
    stdout='✓ regression completed\n{"conclusion":"success"}\n'.encode('utf-8')
    stderr='独立复现'.encode('utf-8')
    decoded,original=decode_command_output(stdout,stderr)
    assert decoded.startswith('✓ regression completed')
    assert original==stdout+b'\n'+stderr
    assert decode_command_output(None,None)==('',b'\n')
