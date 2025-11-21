#!/usr/bin/env python3
"""
CVlization wrapper for Krea Realtime WebSocket server using official SDK.
Provides real-time video generation via WebSocket streaming.
"""
import os
import sys
import argparse

# Ensure SDK is in path
sys.path.insert(0, "/opt/krea-sdk")

def main():
    parser = argparse.ArgumentParser(
        description="Start Krea Realtime WebSocket server for real-time video generation"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind server to (default: 8000)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/self_forcing_server_14b.yaml",
        help="Path to SDK config file (relative to /opt/krea-sdk)"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for better performance (slower startup)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    # Set environment variables for SDK
    os.environ["MODEL_FOLDER"] = os.environ.get("MODEL_FOLDER", "/root/.cache/huggingface/wan_models")
    os.environ["CONFIG"] = args.config
    os.environ["DO_COMPILE"] = "true" if args.compile else "false"

    # Change to SDK directory for config access
    os.chdir("/opt/krea-sdk")

    print("=" * 80)
    print("Krea Realtime WebSocket Server")
    print("=" * 80)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Config: {args.config}")
    print(f"Torch Compile: {args.compile}")
    print(f"Model Folder: {os.environ['MODEL_FOLDER']}")
    print("=" * 80)
    print()
    print("🔄 Starting server (model loading will occur on first request)...")
    print(f"📡 Web UI: http://{args.host}:{args.port}/")
    print(f"🏥 Health: http://{args.host}:{args.port}/health")
    print(f"🔌 WebSocket: ws://{args.host}:{args.port}/ws")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 80)
    print()

    # Import and run uvicorn
    import uvicorn

    # Import the FastAPI app from release_server
    from release_server import app

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
