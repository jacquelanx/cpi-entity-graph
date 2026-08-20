"""
`demo` -- the libraries behind the sample-transcript HTML reports.

Package marker. `cases.py` simulates a perfect detector from `samples/gold/` and
runs the real pipeline; `render/` turns the result into HTML. Both report scripts
in `scripts/` are thin wrappers over these two.

Nothing in `graph/` depends on this package.
"""
