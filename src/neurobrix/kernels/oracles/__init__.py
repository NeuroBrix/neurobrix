"""Float64 references for the autotune correctness screen.

Pure numpy, no torch, no Triton: a reference that shares an implementation with
the thing it checks is not a reference. Each module here answers one kernel
family from its live arguments and says nothing otherwise.
"""
