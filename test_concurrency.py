"""
test_concurrency.py — 20-Session Concurrency Benchmark for Piper TTS Production Server

Usage:
    python test_concurrency.py --url http://localhost:5000/synthesize --concurrency 20
    python test_concurrency.py --url https://<pod_id>-5000.proxy.runpod.net/synthesize --concurrency 20
"""

import argparse
import asyncio
import statistics
import time
from typing import Dict, List
import httpx

TEST_PROMPTS = [
    {"text": "नमस्ते, आपका नापतोल का ऑर्डर कन्फर्म करने के लिए धन्यवाद।", "voice": "hi"},
    {"text": "Hello, thank you for confirming your order with Naaptol.", "voice": "en"},
    {"text": "നമസ്കാരം, നിങ്ങളുടെ ഓർഡർ സ്ഥിരീകരിച്ചതിന് നന്ദി.", "voice": "ml"},
    {"text": "నమస్కారం, మీ ఆర్డర్ నిర్ధారించినందుకు ధన్యవాదాలు.", "voice": "te"},
]


async def send_synthesis_request(
    client: httpx.AsyncClient,
    url: str,
    req_id: int,
    payload: Dict[str, str],
) -> Dict:
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, timeout=20.0)
        dur_ms = (time.perf_counter() - t0) * 1000
        size = len(resp.content)
        success = resp.status_code == 200
        return {
            "req_id": req_id,
            "status": resp.status_code,
            "latency_ms": dur_ms,
            "size_bytes": size,
            "voice": payload.get("voice"),
            "success": success,
        }
    except Exception as e:
        dur_ms = (time.perf_counter() - t0) * 1000
        return {
            "req_id": req_id,
            "status": 0,
            "latency_ms": dur_ms,
            "size_bytes": 0,
            "voice": payload.get("voice"),
            "success": False,
            "error": str(e),
        }


async def run_benchmark(url: str, concurrency: int = 20):
    print("=" * 65)
    print(f"PIPER TTS CONCURRENCY BENCHMARK (Concurrency = {concurrency})")
    print(f"Target URL: {url}")
    print("=" * 65)

    # First check health
    health_url = url.replace("/synthesize", "/health")
    async with httpx.AsyncClient() as client:
        try:
            h = await client.get(health_url, timeout=5.0)
            if h.status_code == 200:
                data = h.json()
                print(f"Health Check: OK (CUDA={data.get('cuda')}, Loaded Voices: {data.get('loaded_voices')})")
            else:
                print(f"Health Check Warning: Status {h.status_code}")
        except Exception as e:
            print(f"Notice: Health check at {health_url} returned error: {e}")

    print(f"\nDispatching {concurrency} requests simultaneously...")
    limits = httpx.Limits(max_connections=concurrency + 5, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = []
        for i in range(concurrency):
            prompt_item = TEST_PROMPTS[i % len(TEST_PROMPTS)]
            payload = {"text": prompt_item["text"], "voice": prompt_item["voice"]}
            tasks.append(send_synthesis_request(client, url, i + 1, payload))

        t_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time_ms = (time.perf_counter() - t_start) * 1000

    latencies = [r["latency_ms"] for r in results if r["success"]]
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print("\nIndividual Request Results:")
    print("-" * 65)
    for r in results:
        status_str = f"HTTP {r['status']}" if r["success"] else f"FAILED ({r.get('error', '')})"
        print(f"Request #{r['req_id']:02d} [{r['voice'].upper()}]: {status_str} | Latency: {r['latency_ms']:.1f}ms | Audio: {r['size_bytes'] / 1024:.1f} KB")

    print("=" * 65)
    print("SUMMARY STATS:")
    print(f"Total Requests:      {concurrency}")
    print(f"Successful:          {successful} / {concurrency}")
    print(f"Failed:              {failed}")
    print(f"Total Wall Time:     {total_time_ms:.1f} ms")
    if latencies:
        print(f"Min Latency:         {min(latencies):.1f} ms")
        print(f"Max Latency:         {max(latencies):.1f} ms")
        print(f"Average Latency:     {statistics.mean(latencies):.1f} ms")
        print(f"Median Latency (p50):{statistics.median(latencies):.1f} ms")
        throughput = (successful / (total_time_ms / 1000.0))
        print(f"Throughput:          {throughput:.1f} requests/sec")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piper TTS Concurrency Tester")
    parser.add_argument(
        "--url",
        default="http://localhost:5000/synthesize",
        help="Full synthesis endpoint URL",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of concurrent requests (default: 20)",
    )
    args = parser.parse_args()

    asyncio.run(run_benchmark(url=args.url, concurrency=args.concurrency))
