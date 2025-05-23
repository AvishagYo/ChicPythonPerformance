import asyncio
import yappi

async def foo():
    await asyncio.sleep(1.0)
    await baz()
    await asyncio.sleep(0.5)

async def bar():
    await asyncio.sleep(2.0)

async def baz():
    await asyncio.sleep(1.0)

yappi.set_clock_type("WALL")
with yappi.run():
    asyncio.run(foo())
    asyncio.run(bar())
yappi.get_func_stats().print_all()
