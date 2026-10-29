"""
Keep-Alive Health Check Batch Worker.
Pings the application /health endpoint every 14 minutes (840 seconds)
to prevent cloud hosting providers like Render (Free Tier) from sleeping.

Usage:
    # Run continuously (every 14 minutes by default):
    python tools/keep_alive.py --url https://your-app.onrender.com

    # Run as a one-time check (useful for external cron jobs or GitHub Actions):
    python tools/keep_alive.py --url https://your-app.onrender.com --once

    # Custom interval (in seconds):
    python tools/keep_alive.py --url https://your-app.onrender.com --interval 600
"""

import os
import sys
import time
import argparse
import urllib.request
import urllib.error
from datetime import datetime


def ping_health(base_url: str, timeout_seconds: int = 30) -> bool:
    """Pings the /health endpoint of the target service."""
    clean_url = base_url.rstrip("/")
    if not clean_url.endswith("/health"):
        health_url = f"{clean_url}/health"
    else:
        health_url = clean_url

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_time = time.perf_counter()

    try:
        req = urllib.request.Request(
            health_url,
            headers={"User-Agent": "Render-KeepAlive-Worker/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            latency_ms = (time.perf_counter() - start_time) * 1000
            status_code = response.getcode()
            print(f"[{timestamp}] ✅ Health Ping OK | HTTP {status_code} | Latency: {latency_ms:.1f}ms | URL: {health_url}")
            return True

    except urllib.error.HTTPError as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        print(f"[{timestamp}] ⚠️ Health Ping HTTP Error: {e.code} ({e.reason}) | Latency: {latency_ms:.1f}ms | URL: {health_url}", file=sys.stderr)
        return False

    except urllib.error.URLError as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        print(f"[{timestamp}] ❌ Health Ping Failed: {e.reason} | URL: {health_url}", file=sys.stderr)
        return False

    except Exception as e:
        print(f"[{timestamp}] ❌ Unexpected error pinging {health_url}: {e}", file=sys.stderr)
        return False


def run_batch_worker(base_url: str, interval_seconds: int, run_once: bool = False):
    """Executes the keep-alive loop every interval_seconds."""
    print("=" * 65)
    print("🚀 Banking Compliance RAG - Cloud Keep-Alive Batch Process")
    print(f"Target URL:        {base_url}")
    print(f"Ping Interval:     {interval_seconds} seconds ({interval_seconds / 60:.1f} minutes)")
    print(f"Mode:              {'Single Execution (--once)' if run_once else 'Continuous Keep-Alive Loop'}")
    print("=" * 65)

    if run_once:
        success = ping_health(base_url)
        sys.exit(0 if success else 1)

    while True:
        ping_health(base_url)
        print(f"Sleeping for {interval_seconds} seconds until next health ping...\n")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Keep-Alive 14-Minute Batch Process")
    parser.add_argument(
        "--url",
        type=str,
        default=os.getenv("RENDER_APP_URL", "http://localhost:8000"),
        help="Target base URL of the deployed application (e.g. https://my-app.onrender.com)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=int(os.getenv("PING_INTERVAL_SECONDS", "840")),  # 14 minutes = 840 seconds
        help="Interval between health checks in seconds (default: 840s = 14m)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single health check ping and exit (exit code 0 on success, 1 on error)"
    )

    args = parser.parse_args()
    try:
        run_batch_worker(args.url, args.interval, args.once)
    except KeyboardInterrupt:
        print("\nKeep-Alive worker stopped by user.")
        sys.exit(0)
