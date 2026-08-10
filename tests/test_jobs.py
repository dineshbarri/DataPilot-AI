import threading

from datapilot.jobs import JobManager


def test_job_completes_and_exposes_result():
    manager = JobManager(workers=1)
    job = manager.submit(lambda: {"answer": 42})
    job.future.result(timeout=5)
    assert manager.get(job.job_id).public()["result"] == {"answer": 42}
    assert job.status == "completed"


def test_job_failure_is_sanitized():
    manager = JobManager(workers=1)
    job = manager.submit(lambda: (_ for _ in ()).throw(RuntimeError("secret detail")))
    job.future.result(timeout=5)
    assert job.status == "failed"
    assert "secret detail" not in job.error


def test_queued_job_can_be_cancelled():
    manager = JobManager(workers=1)
    gate = threading.Event()
    running = manager.submit(lambda: gate.wait(2))
    queued = manager.submit(lambda: "never")
    assert manager.cancel(queued.job_id)
    gate.set()
    running.future.result(timeout=5)
