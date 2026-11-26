import subprocess
import sys


def test_pyplot_is_loaded_only_when_drawing(tmp_path):
    code = """
import sys
from critmc import plotting
assert "matplotlib.pyplot" not in sys.modules
fig, ax = plotting.figure()
ax.plot([1, 2], [3, 4])
plotting.save(fig, sys.argv[1])
import matplotlib
assert matplotlib.get_backend().lower() == "agg"
"""
    png = tmp_path / "figure.png"
    subprocess.run([sys.executable, "-c", code, str(png)], check=True)
    assert png.stat().st_size > 0
