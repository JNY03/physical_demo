import argparse
import asyncio


async def relay(reader, writer, target_host, target_port):
    try:
        target_reader, target_writer = await asyncio.open_connection(target_host, target_port)
        await asyncio.gather(
            pipe(reader, target_writer),
            pipe(target_reader, writer),
        )
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def main(args):
    server = await asyncio.start_server(
        lambda r, w: relay(r, w, args.target_host, args.target_port),
        args.listen_host,
        args.listen_port,
    )
    addresses = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    print(f"Pi3 proxy listening on {addresses} -> {args.target_host}:{args.target_port}", flush=True)
    async with server:
        await server.serve_forever()


parser = argparse.ArgumentParser()
parser.add_argument("--listen-host", default="0.0.0.0")
parser.add_argument("--listen-port", type=int, default=18890)
parser.add_argument("--target-host", default="100.85.243.54")
parser.add_argument("--target-port", type=int, default=8890)
asyncio.run(main(parser.parse_args()))