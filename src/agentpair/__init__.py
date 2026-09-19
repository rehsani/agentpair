"""Implementer and reviewer agent pair with an execution-backed dispute resolver.

One agent implements a task and writes its tests. A second agent, a different model,
reviews the result. The implementer applies the findings it agrees with and returns the
rest as disputes. A dispute that can be settled by running something is settled that way,
never by an opinion about what the code should do, and the probe written to settle it is
retained beside the run so the resolution stays auditable.
"""

__version__ = "0.1.0"
