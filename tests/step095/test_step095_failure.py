import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('step095_acceptance', ROOT / 'benchmarks/check_step095_acceptance.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize('case', runner.CASES)
def test_failure_and_cleanup(case, tmp_path):
    value = runner.execute(case, tmp_path / f'{case}.json', 120)
    assert value['passed'], Path(value['log']).read_text()[-15000:]


def test_shutdown_interrupts_full_shared_memory_queue():
    import threading
    import time
    from minivllm.distributed.shm_broadcast import MessageQueue
    writer = MessageQueue(1, 1, max_chunk_bytes=1024, max_chunks=1)
    reader = MessageQueue.create_from_handle(writer.export_handle(), 0)
    errors = []
    try:
        writer.enqueue('first')
        def blocked_write():
            try:
                writer.enqueue('second')
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=blocked_write)
        thread.start()
        time.sleep(0.1)
        assert thread.is_alive()
        writer.shutdown()
        thread.join(2)
        assert not thread.is_alive()
        assert len(errors) == 1 and '关闭' in str(errors[0])
    finally:
        reader.close()
        writer.close()
