"""CLI tests.

The flag-placement tests exist because argparse's default behaviour (globals
must precede the subcommand) is the opposite of what people type, and the
README's own examples were broken by it.
"""

import pytest

from alpha.cli import main


@pytest.mark.parametrize("argv", [
    ["sectors", "--no-color"],
    ["expiries", "--no-color"],
    ["options", "--symbols", "NIFTY", "--no-color"],
    ["equity", "--top", "2", "--no-color"],
])
def test_subcommands_run(argv, capsys):
    assert main(argv) == 0
    assert capsys.readouterr().out.strip()


def test_flags_work_after_the_subcommand(capsys):
    """`alpha options --explain` -- the form the README documents."""
    assert main(["options", "--symbols", "NIFTY", "--explain", "--no-color"]) == 0
    assert "Why" in capsys.readouterr().out


def test_flags_work_before_the_subcommand(capsys):
    assert main(["--explain", "--no-color", "options", "--symbols", "NIFTY"]) == 0
    assert "Why" in capsys.readouterr().out


def test_as_of_is_honoured_in_either_position(capsys):
    main(["--no-color", "--as-of", "2026-09-01", "expiries"])
    before = capsys.readouterr().out
    main(["expiries", "--no-color", "--as-of", "2026-09-01"])
    assert "01-Sep-2026" in before
    assert "01-Sep-2026" in capsys.readouterr().out


def test_explain_prints_factor_details(capsys):
    main(["options", "--symbols", "BANKNIFTY", "--explain", "--no-color"])
    out = capsys.readouterr().out
    assert "[context]" in out          # weight-0 factors are labelled as such
    assert "Implied vs realised volatility" in out
    assert "Days to expiry" in out


def test_simulated_data_is_announced(capsys):
    main(["sectors", "--no-color"])
    assert "Simulated data" in capsys.readouterr().out


def test_disclaimer_is_always_printed(capsys):
    main(["sectors", "--no-color"])
    assert "not investment advice" in capsys.readouterr().out


def test_unknown_subcommand_exits_nonzero():
    with pytest.raises(SystemExit) as e:
        main(["nonsense"])
    assert e.value.code != 0
