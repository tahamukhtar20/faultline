"""Errors that mean the sandbox failed, so a run says nothing about the agent's code."""


class DockerError(RuntimeError):
    pass


class ControlChannelError(RuntimeError):
    pass
