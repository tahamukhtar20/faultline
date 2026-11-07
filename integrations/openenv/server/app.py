from openenv.core.env_server.http_server import create_app

try:
    from ..models import FaultlineAction, FaultlineObservation
    from .environment import FaultlineEnvironment
except ImportError:
    from models import FaultlineAction, FaultlineObservation
    from server.environment import FaultlineEnvironment

app = create_app(
    FaultlineEnvironment,
    FaultlineAction,
    FaultlineObservation,
    env_name="faultline_env",
    max_concurrent_envs=2,
)


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
