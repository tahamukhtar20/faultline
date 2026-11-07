"""Scores a workspace inside one disposable container, for frameworks that own the sandbox.

Benchmark frameworks give a grader a single container and no Docker. So here the service runs as
plain processes instead, with every process under its own unprivileged user. This package and what
it imports must stay free of Docker and of the task generator, because exported tasks bundle it.
"""
