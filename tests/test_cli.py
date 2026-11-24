"""The command line: arguments are checked before anything runs."""

import numpy as np
import pytest

from critmc.cli import main

FAST = ["--L", "8", "--therm", "50", "--meas", "400", "--seed", "3"]


def run(capsys, *argv):
    assert main(list(argv)) == 0
    return capsys.readouterr().out


def rejected(capsys, *argv):
    """The usage error printed when the arguments are refused."""
    with pytest.raises(SystemExit) as info:
        main(list(argv))
    assert info.value.code == 2
    return capsys.readouterr().err


def test_simulate_prints_a_summary(capsys):
    out = run(capsys, "simulate", *FAST, "--T", "2.5")
    for name in ("<e>", "<absm>", "<m2>", "U4"):
        assert name in out


def test_temperature_and_beta_give_the_same_run(capsys):
    def table(*argv):
        out = run(capsys, "simulate", *FAST, *argv)
        return [line for line in out.splitlines() if "throughput" not in line]

    assert len(table("--T", "2.5")) == 8
    assert table("--T", "2.5") == table("--beta", "0.4")


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_temperature_must_be_positive(capsys, value):
    err = rejected(capsys, "simulate", *FAST, f"--T={value}")
    assert "argument --T: must be a positive number" in err
    err = rejected(capsys, "exact", "thermo", f"--T={value}")
    assert "argument --T: must be a positive number" in err


def test_beta_must_not_be_negative(capsys):
    err = rejected(capsys, "simulate", *FAST, "--beta=-0.3")
    assert "argument --beta: must be a non-negative number" in err
    assert "<e>" in run(capsys, "simulate", *FAST, "--beta", "0")


@pytest.mark.parametrize(
    ("model", "option", "owner"),
    [
        ("potts", "--field=0.5", "ising"),
        ("potts", "--n=7", "on"),
        ("xy", "--q=5", "potts"),
        ("xy", "--n=3", "on"),
        ("heisenberg", "--n=2", "on"),
        ("ising", "--q=4", "potts"),
        ("on", "--field=0.1", "ising"),
    ],
)
def test_options_of_another_model_are_refused(capsys, model, option, owner):
    err = rejected(capsys, "simulate", *FAST, "--model", model, "--T", "1.0", option)
    name = option.split("=")[0]
    assert f"{name} applies to --model {owner} only, not to --model {model}" in err


def test_model_options_still_reach_their_model(capsys):
    updates = ["--updates", "metropolis"]
    plain = run(capsys, "simulate", *FAST, *updates, "--T", "2.5")
    field = run(capsys, "simulate", *FAST, *updates, "--T", "2.5", "--field", "0.5")
    assert plain != field
    q3 = run(capsys, "simulate", *FAST, "--model", "potts", "--T", "1.0")
    q5 = run(capsys, "simulate", *FAST, "--model", "potts", "--T", "1.0", "--q", "5")
    assert "Potts(q=3)" in q3 and "Potts(q=5)" in q5
    o3 = run(capsys, "simulate", *FAST, "--model", "on", "--T", "1.0", "--n", "3")
    heis = run(capsys, "simulate", *FAST, "--model", "heisenberg", "--T", "1.0")
    assert "O(3)" in o3 and "O(3)" in heis


def test_exact_zeros_from_a_large_size_alone(capsys):
    out = run(capsys, "exact", "zeros", "32")
    row = out.splitlines()[-1].split()
    assert row[0] == "32"
    assert float(row[3]) == pytest.approx(2.24064823e-2, rel=1e-8)


BAD_ARGUMENTS = [
    # simulate: sizes, counts and the seed
    (["simulate", "--L=1"], "argument --L: must be an integer >= 2, got 1"),
    (["simulate", "--L=-4"], "argument --L: must be an integer >= 2"),
    (["simulate", "--L=8.5"], "argument --L: must be an integer >= 2"),
    (["simulate", "--meas=0"], "argument --meas: must be an integer >= 2, got 0"),
    (["simulate", "--meas=1"], "argument --meas: must be an integer >= 2"),
    (["simulate", "--therm=-3"], "argument --therm: must be an integer >= 0"),
    (["simulate", "--seed=-1"], "argument --seed: must be an integer >= 0"),
    (["simulate", "--field=nan"], "argument --field: must be a finite number"),
    (["simulate", "--field=inf"], "argument --field: must be a finite number"),
    (["simulate", "--T=warm"], "argument --T: must be a positive number, got warm"),
    # simulate: what only the model can judge
    (["simulate", *FAST, "--updates", "foo"], "unknown updates ['foo']"),
    (["simulate", *FAST, "--updates", "0*wolff"], "must be a positive integer in '0*wolff'"),
    (["simulate", *FAST, "--updates=-2*wolff"], "must be a positive integer in '-2*wolff'"),
    (["simulate", *FAST, "--updates", "x*wolff"], "must be a positive integer in 'x*wolff'"),
    (["simulate", *FAST, "--updates", "+"], "empty update schedule"),
    (["simulate", *FAST, "--updates", "overrelax"], "Ising does not support updates"),
    (["simulate", *FAST, "--field", "0.5"], "cluster updates require zero field h"),
    (["simulate", *FAST, "--model", "potts", "--q", "1"], "q must be in [2, 127], got 1"),
    (["simulate", *FAST, "--model", "potts", "--q", "300"], "q must be in [2, 127], got 300"),
    (["simulate", *FAST, "--model", "on", "--n", "1"], "n must be at least 2, got 1"),
    (["simulate", *FAST, "--model", "on", "--n", "0"], "n must be at least 2, got 0"),
    # exact
    (["exact", "thermo", "--T", "2", "--L", "1"], "must be 0 (thermodynamic limit) or an integer"),
    (["exact", "thermo", "--T", "2", "--L=-4"], "must be 0 (thermodynamic limit) or an integer"),
    (["exact", "dos", "--L", "5"], "argument --L: must be an even integer of at least 2, got 5"),
    (["exact", "dos", "--L", "0"], "argument --L: must be an even integer of at least 2"),
    (["exact", "dos", "--L=-2"], "argument --L: must be an even integer of at least 2"),
    (["exact", "zeros", "5"], "argument sizes: must be an even integer of at least 4, got 5"),
    (["exact", "zeros", "8", "2"], "argument sizes: must be an even integer of at least 4"),
    (["exact", "zeros", "0"], "argument sizes: must be an even integer of at least 4"),
    # bench
    (["bench", "--L", "1"], "argument --L: must be an integer >= 2"),
    (["bench", "--meas", "0"], "argument --meas: must be an integer >= 1"),
]


@pytest.mark.parametrize(("argv", "message"), BAD_ARGUMENTS, ids=lambda v: " ".join(v[:4]))
def test_bad_arguments_are_usage_errors(capsys, argv, message):
    assert message in rejected(capsys, *argv)


def failed(capsys, *argv):
    """The one-line error of a command that could not read or write a file."""
    assert main(list(argv)) == 1
    err = capsys.readouterr().err
    assert err.startswith("critmc: error: ") and err.count("\n") == 1
    return err


def test_the_smallest_arguments_still_run(capsys):
    out = run(capsys, "simulate", "--L", "2", "--therm", "0", "--meas", "2", "--seed", "0")
    assert "L=2" in out and "n=2" in out
    assert "Potts(q=2)" in run(capsys, "simulate", *FAST, "--model", "potts", "--q", "2")
    assert "O(4)" in run(capsys, "simulate", *FAST, "--model", "on", "--n", "4", "--T", "1")
    assert "Upsilon" in run(capsys, "simulate", *FAST, "--model", "xy", "--updates", "overrelax")
    assert len(run(capsys, "exact", "thermo", "--T", "2", "--L", "2").splitlines()) == 2
    assert len(run(capsys, "exact", "thermo", "--T", "2").splitlines()) == 2
    assert run(capsys, "exact", "dos", "--L", "2").split() == ["-8", "2", "0", "12", "8", "2"]


def test_analyze_reads_what_simulate_saved(capsys, tmp_path):
    saved = tmp_path / "new" / "run.npz"
    out = run(capsys, "simulate", *FAST, "--T", "2.5", "--out", str(saved))
    assert f"saved to {saved}" in out
    table = [line for line in out.splitlines() if "throughput" not in line and "saved" not in line]
    again = run(capsys, "analyze", str(saved))
    assert [line for line in again.splitlines() if "throughput" not in line] == table


def test_analyze_refuses_files_it_cannot_read(capsys, tmp_path):
    assert "No such file" in failed(capsys, "analyze", str(tmp_path / "missing.npz"))
    text = tmp_path / "text.npz"
    text.write_text("not an archive")
    assert f"{text} is not an .npz file" in failed(capsys, "analyze", str(text))
    empty = tmp_path / "empty.npz"
    empty.touch()
    assert f"{empty} is not an .npz file" in failed(capsys, "analyze", str(empty))
    other = tmp_path / "other.npz"
    np.savez(other, a=np.arange(5))
    assert f"{other} is not a time series saved by critmc" in failed(capsys, "analyze", str(other))


def test_an_output_that_cannot_be_written_stops_the_run_first(capsys, tmp_path, monkeypatch):
    from critmc import cli

    def never(*args, **kwargs):
        raise AssertionError("the run started")

    monkeypatch.setattr(cli.Simulation, "thermalize", never)
    assert "cannot write" in failed(capsys, "simulate", *FAST, "--out", str(tmp_path))
    blocked = tmp_path / "file"
    blocked.touch()
    failed(capsys, "simulate", *FAST, "--out", str(blocked / "run.npz"))
    failed(capsys, "exact", "dos", "--L", "4", "--out", str(blocked / "dos.txt"))
