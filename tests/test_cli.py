"""The command line: arguments are checked before anything runs."""

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
    with pytest.raises(ValueError, match="even and at least 4"):
        main(["exact", "zeros", "5"])
