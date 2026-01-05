import asyncio
from bleak import BleakScanner

async def scan():
    print("スキャン中...")
    devices = await BleakScanner.discover(timeout=10.0)
    for d in devices:
        print(f"名前: {d.name}, アドレス: {d.address}, RSSI: {d.rssi}")

asyncio.run(scan())
