from scheduler.batching import build_batch
from scheduler.admission import AdmissionQueue
from scheduler.scheduler import Request
def test_running_decode_precedes_chunked_prefill():
    run=Request('r',0,2,0,0); wait=Request('w',10,2,0,0); q=AdmissionQueue();q.push(wait)
    batch=build_batch([run],q,5,2)
    assert [(x.request.id,x.kind,x.tokens) for x in batch]==[('r','decode',1),('w','prefill',4)]
