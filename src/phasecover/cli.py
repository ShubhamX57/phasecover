"""Command-line interface: ``phasecover simulate`` and ``phasecover analyze``."""

import argparse
import sys

import astropy.units as u
import numpy as np

from .io import read_events, write_events
from .search import folded_profile, modulation_test
from .simulate import leo_gtis, simulate_events


def _simulate(args):
    rng = np.random.default_rng(args.seed)
    gtis = leo_gtis(args.duration * u.hour, orbit_period=args.orbit * u.min, rng=rng)
    times = simulate_events(
        gtis,
        rate=args.rate,
        period=args.period * u.hour,
        pulsed_fraction=args.pulsed_fraction,
        rng=rng,
    )
    write_events(args.output, times, gtis, TELESCOP="SIMULATED", OBJECT="TEST-SOURCE")
    print(f"Wrote {times.size} events in {len(gtis)} GTIs to {args.output}")


def _analyze(args):
    events = read_events(args.events)
    period = args.period * u.hour
    result = modulation_test(events.times, events.gtis, period)
    print(f"File                  : {args.events}")
    print(f"Start                 : {events.start_time.utc.iso} UTC")
    print(f"Exposure              : {events.exposure.to(u.ks):.2f}")
    print(result.summary())
    verdict = "modulation detected" if result.is_significant(args.threshold) else "no significant modulation"
    print(f"Verdict (corrected)   : {verdict} (threshold fpp < {args.threshold:g})")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        phase, rate, err, exposure = folded_profile(events.times, events.gtis, period, args.nbins)
        counts = rate * exposure
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 5))
        ax1.step(phase, counts, where="mid")
        ax1.set_ylabel("counts per bin\n(not corrected)")
        ax2.errorbar(phase, rate, err, fmt="o")
        ax2.set_ylabel("rate (counts / s)\n(exposure-corrected)")
        ax2.set_xlabel("phase")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Saved folded profile to {args.plot}")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="phasecover",
        description="Exposure-corrected modulation searches for gappy event data.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="write a simulated OGIP event file")
    sim.add_argument("output", help="output FITS file")
    sim.add_argument("--duration", type=float, default=30.0, help="span in hours (default 30)")
    sim.add_argument("--orbit", type=float, default=96.0, help="satellite orbit in minutes (default 96)")
    sim.add_argument("--rate", type=float, default=2.0, help="mean count rate, counts/s (default 2)")
    sim.add_argument("--period", type=float, default=5.0, help="source period in hours (default 5)")
    sim.add_argument("--pulsed-fraction", type=float, default=0.0, help="modulation amplitude 0-1 (default 0)")
    sim.add_argument("--seed", type=int, default=None, help="random seed")
    sim.set_defaults(func=_simulate)

    ana = sub.add_parser("analyze", help="test an event file for modulation at a given period")
    ana.add_argument("events", help="OGIP event FITS file")
    ana.add_argument("--period", type=float, required=True, help="trial period in hours")
    ana.add_argument("--threshold", type=float, default=1e-3, help="false-positive probability threshold")
    ana.add_argument("--nbins", type=int, default=16, help="phase bins for --plot")
    ana.add_argument("--plot", help="save a folded-profile figure to this file")
    ana.set_defaults(func=_analyze)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
