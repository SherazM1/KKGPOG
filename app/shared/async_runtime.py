"""Run browser jobs independently of Streamlit's Windows event-loop policy."""
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor


def run_browser_job(coroutine_factory):
    """Use a worker-owned subprocess-capable loop without changing app policy."""
    def run():
        loop_factory = asyncio.ProactorEventLoop if sys.platform == "win32" else asyncio.new_event_loop
        with asyncio.Runner(loop_factory=loop_factory) as runner:
            return runner.run(coroutine_factory())

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="sams-browser") as executor:
        return executor.submit(run).result()
