import asyncio
import unittest
from paste_repro.invocation import Invocation
from paste_repro.live_broker import LiveToolBroker
from paste_repro.live_executor import SyncToolMapExecutor
from paste_repro.resource_accounting import summarize_resources


class ResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_thread_cpu_and_late_cleanup_are_charged_once(self):
        class Tool:
            def call(self,args):
                return sum(i*i for i in range(20000))
        async with SyncToolMapExecutor({'read':Tool()},thread_workers=2) as executor:
            broker=LiveToolBroker(executor,max_workers=2,max_speculative_workers=1)
            await broker.speculate(Invocation('read',{'value':'unused'}),session_id='task')
            await broker.authoritative(Invocation('read',{'value':'used'}),session_id='task')
            await broker.close()
            records=broker.tool_records()
            cost=broker.resource_summary()['task']
            assert cost['physical_calls']==2
            assert cost['cpu_core_s']>0
            assert cost['unused_speculative_calls']==1
            assert cost['cleanup_complete']
            assert cost['http_requests'] is None
            assert len({r['job_id'] for r in records})==2

    async def test_unknown_cost_never_becomes_zero(self):
        summary=summarize_resources([{'admitted':True,'started_at':1,'service_s':1,'cleanup_at':2}],now=2)
        assert summary['cpu_core_s'] is None
        assert summary['http_requests'] is None

    async def test_payload_retention_integrates_overlap(self):
        rows=[{'admitted':True,'finished_at':1,'cleanup_at':3,'retained_result_bytes':10},
              {'admitted':True,'finished_at':2,'cleanup_at':4,'retained_result_bytes':20}]
        summary=summarize_resources(rows,now=4)
        assert summary['private_result_peak_bytes']==30
        assert summary['private_result_byte_s']==60

    async def test_cancelled_http_keeps_redirect_and_received_body_cost(self):
        from aiohttp import web
        from paste_repro.live_executor import WikipediaLiveExecutor
        sent, release = asyncio.Event(), asyncio.Event()

        async def redirect(request):
            raise web.HTTPFound('/stream')

        async def stream(request):
            response = web.StreamResponse()
            await response.prepare(request)
            await response.write(b'x' * 64)
            sent.set()
            await release.wait()
            return response

        app = web.Application()
        app.router.add_get('/redirect', redirect)
        app.router.add_get('/stream', stream)
        server = web.AppRunner(app)
        await server.setup()
        site = web.TCPSite(server, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with WikipediaLiveExecutor() as executor:
                broker = LiveToolBroker(executor, max_workers=2, max_speculative_workers=1)
                await broker.speculate(Invocation('visit', {'url': f'http://127.0.0.1:{port}/redirect'}), session_id='task')
                await asyncio.wait_for(sent.wait(), 5)
                await asyncio.sleep(0.02)
                await broker.close()
                summary = broker.resource_summary()['task']
                assert summary['http_requests'] == 2
                assert summary['http_response_body_bytes'] == 64
                assert summary['unused_speculative_calls'] == 1
                assert summary['cleanup_complete']
        finally:
            release.set()
            await server.cleanup()
