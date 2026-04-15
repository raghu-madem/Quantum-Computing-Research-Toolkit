#!/usr/bin/env python3
"""
Quantum Computing Research Toolkit — CLI
==========================================
Interactive command-line interface for all quantum algorithms.

Usage:
    python cli.py --help
    python cli.py bb84 --n-bits 256 --eve
    python cli.py grover --qubits 8 --target 100
    python cli.py teleport --state |+⟩
    python cli.py qaoa --nodes 6 --layers 2
    python cli.py benchmark --all
    python cli.py demo
"""

import argparse
import logging
import math
import sys
import time


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def print_banner() -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        banner = Text()
        banner.append("⚛  ", style="bold cyan")
        banner.append("Quantum Computing Research Toolkit", style="bold white")
        banner.append("  ⚛", style="bold cyan")
        banner.append(
            "\n   BB84 · Grover · Teleportation · QAOA · Benchmarks",
            style="dim",
        )
        console.print(Panel(banner, style="blue", expand=False))
        console.print()
    except ImportError:
        print("\n" + "=" * 55)
        print("  Quantum Computing Research Toolkit")
        print("  BB84 · Grover · Teleportation · QAOA")
        print("=" * 55 + "\n")


# ------------------------------------------------------------------
# Sub-commands
# ------------------------------------------------------------------


def cmd_bb84(args) -> None:
    from quantum_suite.bb84 import BB84Protocol
    from quantum_suite.visualizer import QuantumVisualizer

    viz = QuantumVisualizer(output_dir=args.output_dir)
    print_banner()

    proto = BB84Protocol(
        n_bits=args.n_bits,
        noise_level=args.noise,
        seed=args.seed,
    )
    t0 = time.perf_counter()
    result = proto.run(eve_present=args.eve)
    elapsed = time.perf_counter() - t0

    viz.print_bb84_summary(result, verbose=args.verbose)

    try:
        from rich.console import Console
        Console().print(f"\n[dim]Completed in {elapsed:.2f}s[/dim]")
    except ImportError:
        print(f"\nCompleted in {elapsed:.2f}s")

    if args.plot:
        print("Running QBER analysis sweep...")
        from quantum_suite.benchmarks import QuantumBenchmark
        bench = QuantumBenchmark()
        data = bench.analyze_qber_vs_eve(n_bits=args.n_bits, trials=3)
        path = viz.plot_bb84_qber_analysis(data, save=True)
        if path:
            print(f"Plot saved: {path}")


def cmd_grover(args) -> None:
    from quantum_suite.grover import GroverSearch
    from quantum_suite.visualizer import QuantumVisualizer

    viz = QuantumVisualizer(output_dir=args.output_dir)
    print_banner()

    n_qubits = args.qubits
    target = args.target if args.target is not None else (2**n_qubits) // 3

    grover = GroverSearch(n_qubits=n_qubits, shots=args.shots)
    t0 = time.perf_counter()
    result = grover.search(target=target, n_iterations=args.iterations)
    elapsed = time.perf_counter() - t0

    viz.print_grover_result(result)

    try:
        from rich.console import Console
        Console().print(f"\n[dim]Completed in {elapsed:.2f}s[/dim]")
    except ImportError:
        print(f"\nCompleted in {elapsed:.2f}s")

    if args.plot:
        path = viz.plot_grover_histogram(result, save=True)
        if path:
            print(f"Plot saved: {path}")


def cmd_teleport(args) -> None:
    from quantum_suite.teleportation import QuantumTeleportation
    from quantum_suite.visualizer import QuantumVisualizer

    viz = QuantumVisualizer(output_dir=args.output_dir)
    print_banner()

    teleport = QuantumTeleportation(noise_level=args.noise)

    if args.sweep:
        # Sweep all named states
        try:
            from rich.console import Console
            Console().print("[bold]Running fidelity sweep across all named states...[/bold]")
        except ImportError:
            print("Running fidelity sweep...")

        fidelities = teleport.run_fidelity_sweep()
        for state, f in fidelities.items():
            print(f"  {state:8s}: F = {f:.6f}")

        if args.plot:
            path = viz.plot_teleportation_fidelity(fidelities, save=True)
            if path:
                print(f"\nPlot saved: {path}")
    else:
        state_name = args.state or "|+⟩"
        t0 = time.perf_counter()
        result = teleport.teleport_named_state(state_name)
        elapsed = time.perf_counter() - t0
        viz.print_teleportation_result(result)
        print(f"\nCompleted in {elapsed:.2f}s")


def cmd_qaoa(args) -> None:
    from quantum_suite.qaoa import Graph, QAOAOptimizer
    from quantum_suite.visualizer import QuantumVisualizer

    viz = QuantumVisualizer(output_dir=args.output_dir)
    print_banner()

    # Build graph
    if args.graph_type == "cycle":
        graph = Graph.cycle(args.nodes)
    elif args.graph_type == "complete":
        graph = Graph.complete(args.nodes)
    else:  # random
        graph = Graph.random(args.nodes, edge_prob=0.6, seed=args.seed or 42)

    try:
        from rich.console import Console
        Console().print(
            f"[bold]QAOA Max-Cut:[/bold] {args.nodes}-node {args.graph_type} graph, "
            f"p={args.layers}"
        )
    except ImportError:
        print(f"QAOA Max-Cut: {args.nodes}-node {args.graph_type} graph, p={args.layers}")

    qaoa = QAOAOptimizer(p=args.layers, shots=args.shots, max_iter=args.max_iter)
    t0 = time.perf_counter()
    result = qaoa.solve(graph)
    elapsed = time.perf_counter() - t0

    # Print result
    print(f"\nQAOA Result:")
    print(f"  Best partition: {result.best_partition}")
    print(f"  Cut value:      {result.best_cut_value:.3f}")
    print(f"  Optimal cut:    {result.classical_max_cut:.3f}")
    print(f"  Approx ratio:   {result.approximation_ratio:.4f}")
    print(f"  Optimizer iters:{result.n_optimizer_iterations}")
    print(f"\nCompleted in {elapsed:.2f}s")

    if args.plot:
        path = viz.plot_qaoa_result(result, save=True)
        if path:
            print(f"Plot saved: {path}")


def cmd_benchmark(args) -> None:
    from quantum_suite.benchmarks import QuantumBenchmark
    from quantum_suite.visualizer import QuantumVisualizer

    viz = QuantumVisualizer(output_dir=args.output_dir)
    bench = QuantumBenchmark()
    print_banner()

    if args.all or args.search:
        print("📊  Benchmarking: Grover Search vs Classical Search...")
        report = bench.benchmark_search(sizes=[16, 64, 256, 1024, 4096])
        print(report.print_table())
        if args.plot:
            scaling = bench.analyze_grover_scaling(max_qubits=14)
            path = viz.plot_scaling_comparison(scaling, save=True)
            if path:
                print(f"Scaling plot saved: {path}")

    if args.all or args.maxcut:
        print("\n📊  Benchmarking: QAOA vs Greedy Max-Cut...")
        report = bench.benchmark_maxcut(sizes=[4, 6, 8])
        print(report.print_table())


def cmd_demo(args) -> None:
    """Run a quick demonstration of all algorithms."""
    print_banner()

    try:
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn
        console = Console()
    except ImportError:
        console = None

    demos = [
        ("BB84 Quantum Key Distribution", _demo_bb84),
        ("Grover's Search (6 qubits)", _demo_grover),
        ("Quantum Teleportation", _demo_teleport),
        ("QAOA Max-Cut (5 nodes)", _demo_qaoa),
    ]

    for name, fn in demos:
        if console:
            console.rule(f"[bold cyan]{name}[/bold cyan]")
        else:
            print(f"\n{'='*50}\n  {name}\n{'='*50}")
        try:
            fn()
        except Exception as e:
            if console:
                console.print(f"[red]Error: {e}[/red]")
            else:
                print(f"Error: {e}")


def _demo_bb84():
    from quantum_suite.bb84 import BB84Protocol
    from quantum_suite.visualizer import QuantumVisualizer
    viz = QuantumVisualizer()
    proto = BB84Protocol(n_bits=128, seed=42)
    result = proto.run(eve_present=False)
    viz.print_bb84_summary(result)
    proto_eve = BB84Protocol(n_bits=128, seed=123)
    result_eve = proto_eve.run(eve_present=True)
    try:
        from rich.console import Console
        Console().print("\n[bold red]With Eve:[/bold red]")
    except ImportError:
        print("\nWith Eve:")
    viz.print_bb84_summary(result_eve)


def _demo_grover():
    from quantum_suite.grover import GroverSearch
    from quantum_suite.visualizer import QuantumVisualizer
    viz = QuantumVisualizer()
    grover = GroverSearch(n_qubits=6, shots=2048)
    result = grover.search(target=42)
    viz.print_grover_result(result)


def _demo_teleport():
    from quantum_suite.teleportation import QuantumTeleportation
    from quantum_suite.visualizer import QuantumVisualizer
    viz = QuantumVisualizer()
    teleport = QuantumTeleportation(noise_level=0.0)
    for state_name in ["|+⟩", "|T⟩", "|1⟩"]:
        result = teleport.teleport_named_state(state_name)
        viz.print_teleportation_result(result)


def _demo_qaoa():
    from quantum_suite.qaoa import Graph, QAOAOptimizer
    from quantum_suite.visualizer import QuantumVisualizer
    viz = QuantumVisualizer()
    graph = Graph.cycle(5)
    qaoa = QAOAOptimizer(p=1, shots=2048, max_iter=100)
    result = qaoa.solve(graph)
    print(f"  Partition: {result.best_partition}")
    print(f"  Cut={result.best_cut_value:.2f} / Optimal={result.classical_max_cut:.2f}")
    print(f"  Approx ratio: {result.approximation_ratio:.3f}")


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantum-toolkit",
        description="Quantum Computing Research Toolkit — BB84, Grover, Teleportation, QAOA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output-dir", default="plots", metavar="DIR")

    sub = parser.add_subparsers(dest="command", required=True)

    # BB84
    p_bb84 = sub.add_parser("bb84", help="BB84 Quantum Key Distribution")
    p_bb84.add_argument("--n-bits", type=int, default=256)
    p_bb84.add_argument("--noise", type=float, default=0.0)
    p_bb84.add_argument("--eve", action="store_true", help="Simulate eavesdropper")
    p_bb84.add_argument("--seed", type=int, default=None)
    p_bb84.add_argument("--plot", action="store_true")

    # Grover
    p_grover = sub.add_parser("grover", help="Grover's Search Algorithm")
    p_grover.add_argument("--qubits", type=int, default=6)
    p_grover.add_argument("--target", type=int, default=None)
    p_grover.add_argument("--iterations", type=int, default=None)
    p_grover.add_argument("--shots", type=int, default=2048)
    p_grover.add_argument("--plot", action="store_true")

    # Teleportation
    p_tel = sub.add_parser("teleport", help="Quantum Teleportation")
    p_tel.add_argument("--state", type=str, default="|+⟩",
                       choices=["|0⟩", "|1⟩", "|+⟩", "|-⟩", "|i⟩", "|T⟩", "|Y⟩"])
    p_tel.add_argument("--noise", type=float, default=0.0)
    p_tel.add_argument("--sweep", action="store_true", help="Test all named states")
    p_tel.add_argument("--plot", action="store_true")

    # QAOA
    p_qaoa = sub.add_parser("qaoa", help="QAOA Max-Cut Optimization")
    p_qaoa.add_argument("--nodes", type=int, default=6)
    p_qaoa.add_argument("--layers", type=int, default=2, dest="layers")
    p_qaoa.add_argument("--graph-type", choices=["random", "cycle", "complete"],
                        default="random")
    p_qaoa.add_argument("--shots", type=int, default=4096)
    p_qaoa.add_argument("--max-iter", type=int, default=150)
    p_qaoa.add_argument("--seed", type=int, default=42)
    p_qaoa.add_argument("--plot", action="store_true")

    # Benchmark
    p_bench = sub.add_parser("benchmark", help="Classical vs Quantum benchmarks")
    p_bench.add_argument("--all", action="store_true", help="Run all benchmarks")
    p_bench.add_argument("--search", action="store_true")
    p_bench.add_argument("--maxcut", action="store_true")
    p_bench.add_argument("--plot", action="store_true")

    # Demo
    sub.add_parser("demo", help="Quick demonstration of all algorithms")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    commands = {
        "bb84": cmd_bb84,
        "grover": cmd_grover,
        "teleport": cmd_teleport,
        "qaoa": cmd_qaoa,
        "benchmark": cmd_benchmark,
        "demo": cmd_demo,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        if hasattr(args, "verbose") and args.verbose:
            raise
        print(f"\nError: {e}")
        print("Run with --verbose for traceback.")
        sys.exit(1)


if __name__ == "__main__":
    main()
