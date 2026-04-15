"""
Quantum Visualizer
===================
Rich terminal output and Matplotlib visualizations for all algorithms.
Provides circuit diagrams, probability histograms, Bloch sphere plots,
scaling curves, and QBER analysis charts.
"""

import logging
import math
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Graceful imports — visualizations are optional (no crash if unavailable)
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    logger.warning("matplotlib not found — visual plots disabled")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich import box
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False
    logger.warning("rich not found — falling back to plain text output")


console = Console() if _HAS_RICH else None


class QuantumVisualizer:
    """
    Visualization toolkit for quantum algorithm results.

    Provides:
        - Rich terminal tables and panels
        - Matplotlib probability histograms
        - Bloch sphere state visualization
        - Grover amplitude evolution plots
        - QAOA cost landscape plots
        - Benchmark scaling curves

    Args:
        output_dir (str): Directory to save plots. Default: 'plots/'.
        dpi (int): Plot DPI for saved figures.

    Example:
        >>> viz = QuantumVisualizer(output_dir="./plots")
        >>> viz.plot_grover_result(result, save=True)
        >>> viz.print_bb84_summary(result)
    """

    COLORS = {
        "primary": "#4A90D9",
        "secondary": "#E74C3C",
        "accent": "#2ECC71",
        "warning": "#F39C12",
        "dark": "#2C3E50",
        "light": "#ECF0F1",
        "purple": "#9B59B6",
        "teal": "#1ABC9C",
    }

    def __init__(self, output_dir: str = "plots", dpi: int = 150):
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # BB84 Visualizations
    # ------------------------------------------------------------------

    def print_bb84_summary(self, result, verbose: bool = False) -> None:
        """Print a rich BB84 session summary to the terminal."""
        if not _HAS_RICH:
            self._plain_bb84_summary(result)
            return

        # Header
        status_color = "green" if result.secure else "red"
        status_text = "✅ SECURE" if result.secure else "⚠️  COMPROMISED"
        eve_text = "🚨 EVE DETECTED" if result.eve_detected else "✅ No eavesdropping"

        console.print()
        console.print(
            Panel(
                f"[bold]BB84 Quantum Key Distribution[/bold]\n"
                f"Session ID: [cyan]{result.session_id}[/cyan]",
                style="blue",
                expand=False,
            )
        )

        # Metrics table
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Raw Bits Sent", str(result.raw_key_length))
        table.add_row(
            "Sifted Key Bits",
            f"{result.sifted_key_length} "
            f"({result.sifted_key_length/result.raw_key_length:.0%} efficiency)",
        )
        table.add_row(
            "QBER",
            f"[{'red' if result.qber > 0.11 else 'green'}]{result.qber:.2%}[/]"
            f"  (threshold: 11%)",
        )
        table.add_row("Eavesdropping", f"[{status_color}]{eve_text}[/]")
        table.add_row("Channel Security", f"[bold {status_color}]{status_text}[/bold {status_color}]")
        if result.final_key:
            key_preview = result.final_key[:16] + "..."
            table.add_row("Final Key (SHA-256)", f"[dim]{key_preview}[/dim]")

        console.print(table)

        if verbose:
            console.print(
                f"\n[dim]Bit Sample (first 20): "
                f"Alice={result.alice_bits[:20]}, "
                f"Bob={result.bob_measurements[:20]}[/dim]"
            )

    def plot_bb84_qber_analysis(
        self, analysis_data: dict, save: bool = True
    ) -> Optional[str]:
        """Plot QBER vs Eve interception rate."""
        if not _HAS_MATPLOTLIB:
            logger.warning("matplotlib not available — skipping plot")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(self.COLORS["dark"])
        ax.set_facecolor("#1a252f")

        rates = [r * 100 for r in analysis_data["eve_intercept_rates"]]
        qbers = [q * 100 for q in analysis_data["mean_qbers"]]
        threshold = analysis_data["threshold"] * 100

        ax.plot(rates, qbers, "o-", color=self.COLORS["primary"],
                linewidth=2.5, markersize=8, label="Measured QBER", zorder=5)
        ax.axhline(threshold, color=self.COLORS["secondary"], linestyle="--",
                   linewidth=2, label=f"Security Threshold ({threshold:.0f}%)", zorder=4)
        ax.fill_between(rates, qbers, threshold,
                        where=[q > threshold for q in qbers],
                        alpha=0.2, color=self.COLORS["secondary"],
                        label="Detected as Attack")

        ax.set_xlabel("Eve Interception Rate (%)", color="white", fontsize=12)
        ax.set_ylabel("Quantum Bit Error Rate (%)", color="white", fontsize=12)
        ax.set_title("BB84: QBER vs Eavesdropping Rate", color="white",
                     fontsize=14, fontweight="bold")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#2C3E50", labelcolor="white", fontsize=10)
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.2, color="white")

        plt.tight_layout()
        if save:
            path = os.path.join(self.output_dir, "bb84_qber_analysis.png")
            plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close()
            return path
        plt.show()
        return None

    # ------------------------------------------------------------------
    # Grover Visualizations
    # ------------------------------------------------------------------

    def print_grover_result(self, result) -> None:
        """Print Grover search result to the terminal."""
        if not _HAS_RICH:
            self._plain_grover_result(result)
            return

        found_color = "green" if result.found else "red"
        found_text = "✅ FOUND" if result.found else "❌ NOT FOUND"

        console.print()
        console.print(
            Panel(
                f"[bold]Grover's Quantum Search[/bold]\n"
                f"Search space: [cyan]{result.search_space_size:,}[/cyan] items  "
                f"({result.n_qubits} qubits)",
                style="blue",
                expand=False,
            )
        )

        table = Table(box=box.ROUNDED, header_style="bold cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Target State", str(result.target))
        table.add_row("Measured State", str(result.measured_state))
        table.add_row("Result", f"[bold {found_color}]{found_text}[/bold {found_color}]")
        table.add_row("Success Probability", f"{result.success_probability:.2%}")
        table.add_row("Grover Iterations", str(result.n_iterations))
        table.add_row("Quantum Oracle Calls", str(result.quantum_queries_used))
        table.add_row("Classical Expected Queries", f"{result.classical_queries_expected:,}")
        table.add_row(
            "Quantum Speedup",
            f"[bold green]{result.speedup_factor:.1f}x[/bold green]",
        )

        console.print(table)

    def plot_grover_histogram(self, result, save: bool = True) -> Optional[str]:
        """Plot measurement probability histogram for Grover search."""
        if not _HAS_MATPLOTLIB:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor(self.COLORS["dark"])
        for ax in [ax1, ax2]:
            ax.set_facecolor("#1a252f")

        # Histogram: top 20 states
        top_items = sorted(result.counts.items(), key=lambda x: -x[1])[:20]
        labels = [f"|{int(k,2)}⟩" for k, _ in top_items]
        values = [v / sum(result.counts.values()) for _, v in top_items]
        colors = [
            self.COLORS["accent"] if int(k, 2) == result.measured_state
            else self.COLORS["primary"]
            for k, _ in top_items
        ]

        bars = ax1.bar(range(len(labels)), values, color=colors, edgecolor="white",
                       linewidth=0.5)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha="right", color="white", fontsize=8)
        ax1.set_ylabel("Probability", color="white")
        ax1.set_title("Measurement Probabilities (Top 20)", color="white",
                      fontweight="bold")
        ax1.tick_params(colors="white")
        ax1.spines["bottom"].set_color("white")
        ax1.spines["left"].set_color("white")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.grid(True, axis="y", alpha=0.2, color="white")

        # Theoretical vs actual probability over iterations
        n_qubits = result.n_qubits
        N = result.search_space_size
        iter_range = list(range(0, result.n_iterations * 3 + 1))
        theta = math.asin(1 / math.sqrt(N))
        theoretical = [math.sin((2 * k + 1) * theta) ** 2 for k in iter_range]

        ax2.plot(iter_range, theoretical, color=self.COLORS["primary"],
                 linewidth=2.5, label="Theoretical P(success)")
        ax2.axvline(result.n_iterations, color=self.COLORS["accent"],
                    linestyle="--", linewidth=2,
                    label=f"Optimal k={result.n_iterations}")
        ax2.scatter([result.n_iterations], [result.success_probability],
                    color=self.COLORS["secondary"], s=100, zorder=5,
                    label=f"Measured P={result.success_probability:.2%}")

        ax2.set_xlabel("Number of Grover Iterations", color="white")
        ax2.set_ylabel("P(target state)", color="white")
        ax2.set_title("Success Probability vs Iterations", color="white",
                      fontweight="bold")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(colors="white")
        ax2.legend(facecolor="#2C3E50", labelcolor="white", fontsize=9)
        ax2.spines["bottom"].set_color("white")
        ax2.spines["left"].set_color("white")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(True, alpha=0.2, color="white")

        plt.suptitle(
            f"Grover Search: N={N}, target={result.target}",
            color="white", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, "grover_results.png")
            plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close()
            return path
        plt.show()
        return None

    # ------------------------------------------------------------------
    # Teleportation Visualizations
    # ------------------------------------------------------------------

    def print_teleportation_result(self, result) -> None:
        """Print teleportation fidelity result."""
        if not _HAS_RICH:
            print(f"Teleportation: {result.state_label}, F={result.fidelity:.4f}")
            return

        fidelity_color = "green" if result.fidelity >= 0.95 else "yellow" if result.fidelity >= 0.8 else "red"

        console.print()
        console.print(
            Panel(
                f"[bold]Quantum Teleportation[/bold]\n"
                f"State: [cyan]{result.state_label}[/cyan]",
                style="blue",
                expand=False,
            )
        )

        table = Table(box=box.ROUNDED, header_style="bold cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Input State", result.state_label)
        table.add_row(
            "Fidelity F",
            f"[bold {fidelity_color}]{result.fidelity:.6f}[/bold {fidelity_color}]"
            f"  {'✅ Perfect' if result.fidelity > 0.999 else '✅ High' if result.fidelity >= 0.95 else '⚠️  Degraded'}",
        )
        table.add_row("Alice's Bits (m₀, m₁)", str(result.alice_measurement))
        table.add_row("Bob's Correction", result.correction_applied)
        table.add_row("Noise Level", f"{result.noise_level:.3f}")
        table.add_row("Success", "✅ Yes" if result.success else "❌ No")

        console.print(table)

    def plot_teleportation_fidelity(
        self, fidelity_data: dict[str, float], save: bool = True
    ) -> Optional[str]:
        """Bar chart of teleportation fidelity across named states."""
        if not _HAS_MATPLOTLIB:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(self.COLORS["dark"])
        ax.set_facecolor("#1a252f")

        states = list(fidelity_data.keys())
        fidelities = list(fidelity_data.values())
        colors = [
            self.COLORS["accent"] if f >= 0.99
            else self.COLORS["primary"] if f >= 0.95
            else self.COLORS["warning"]
            for f in fidelities
        ]

        bars = ax.bar(states, fidelities, color=colors, edgecolor="white", linewidth=0.8)
        ax.axhline(1.0, color="white", linestyle="--", alpha=0.4, linewidth=1.5,
                   label="Ideal F=1.0")
        ax.axhline(0.95, color=self.COLORS["warning"], linestyle=":",
                   alpha=0.7, linewidth=1.5, label="Threshold F=0.95")

        for bar, f in zip(bars, fidelities):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{f:.4f}", ha="center", va="bottom", color="white", fontsize=10)

        ax.set_ylim(0, 1.08)
        ax.set_xlabel("Quantum State", color="white", fontsize=12)
        ax.set_ylabel("Teleportation Fidelity F", color="white", fontsize=12)
        ax.set_title("Quantum Teleportation Fidelity — All Named States",
                     color="white", fontsize=14, fontweight="bold")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#2C3E50", labelcolor="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.2, color="white")

        plt.tight_layout()
        if save:
            path = os.path.join(self.output_dir, "teleportation_fidelity.png")
            plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close()
            return path
        plt.show()
        return None

    # ------------------------------------------------------------------
    # Benchmark Visualizations
    # ------------------------------------------------------------------

    def plot_scaling_comparison(
        self, scaling_data: dict, save: bool = True
    ) -> Optional[str]:
        """Plot classical vs Grover query complexity scaling."""
        if not _HAS_MATPLOTLIB:
            return None

        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor(self.COLORS["dark"])
        ax.set_facecolor("#1a252f")

        sizes = scaling_data["problem_sizes"]
        classical = scaling_data["classical_queries"]
        grover = scaling_data["grover_queries"]
        theory = scaling_data["theoretical_grover"]

        ax.loglog(sizes, classical, "o-", color=self.COLORS["secondary"],
                  linewidth=2.5, markersize=7, label="Classical O(N/2)", zorder=5)
        ax.loglog(sizes, grover, "s-", color=self.COLORS["primary"],
                  linewidth=2.5, markersize=7, label="Grover O(√N)", zorder=5)
        ax.loglog(sizes, theory, "--", color=self.COLORS["accent"],
                  linewidth=1.5, alpha=0.7, label="Theoretical √N·π/4")

        ax.fill_between(sizes, classical, grover, alpha=0.15,
                        color=self.COLORS["accent"], label="Quantum Advantage Region")

        ax.set_xlabel("Search Space Size N", color="white", fontsize=12)
        ax.set_ylabel("Oracle Calls (Queries)", color="white", fontsize=12)
        ax.set_title("Query Complexity: Classical Search vs Grover's Algorithm",
                     color="white", fontsize=14, fontweight="bold")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#2C3E50", labelcolor="white", fontsize=10)
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.2, color="white", which="both")

        plt.tight_layout()
        if save:
            path = os.path.join(self.output_dir, "grover_scaling.png")
            plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close()
            return path
        plt.show()
        return None

    def plot_qaoa_result(self, result, save: bool = True) -> Optional[str]:
        """Visualize QAOA optimization result and graph partition."""
        if not _HAS_MATPLOTLIB:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor(self.COLORS["dark"])
        for ax in [ax1, ax2]:
            ax.set_facecolor("#1a252f")

        # Left: top 20 bit-string probabilities
        top = sorted(result.counts.items(), key=lambda x: -x[1])[:15]
        labels = [k for k, _ in top]
        vals = [v / sum(result.counts.values()) for _, v in top]

        best_bit = format(result.best_partition[0], f"0{result.n_qubits if hasattr(result,'n_qubits') else len(result.best_partition)}b")
        colors = [
            self.COLORS["accent"] if k == "".join(map(str, reversed(result.best_partition)))
            else self.COLORS["primary"]
            for k, _ in top
        ]

        ax1.bar(range(len(labels)), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha="right", color="white", fontsize=7)
        ax1.set_ylabel("Probability", color="white")
        ax1.set_title("QAOA Measurement Distribution", color="white", fontweight="bold")
        ax1.tick_params(colors="white")
        ax1.spines["bottom"].set_color("white")
        ax1.spines["left"].set_color("white")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.grid(True, axis="y", alpha=0.2)

        # Right: approximation ratio comparison
        categories = ["QAOA", "Classical\nOptimal", "Approx.\nRatio"]
        values = [
            result.best_cut_value,
            result.classical_max_cut,
            result.approximation_ratio * result.classical_max_cut,
        ]
        bar_colors = [self.COLORS["primary"], self.COLORS["accent"],
                      self.COLORS["warning"]]
        bars = ax2.bar(categories, values, color=bar_colors, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{val:.2f}", ha="center", va="bottom", color="white", fontsize=11,
                     fontweight="bold")

        ax2.set_ylabel("Cut Value", color="white")
        ax2.set_title(
            f"QAOA Max-Cut Result\n"
            f"Approximation Ratio: {result.approximation_ratio:.3f}",
            color="white", fontweight="bold",
        )
        ax2.tick_params(colors="white")
        ax2.spines["bottom"].set_color("white")
        ax2.spines["left"].set_color("white")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(True, axis="y", alpha=0.2)

        plt.suptitle(
            f"QAOA p={result.p_layers}: {result.graph.n_vertices} nodes, "
            f"{len(result.graph.edges)} edges",
            color="white", fontsize=14, fontweight="bold",
        )
        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, "qaoa_result.png")
            plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close()
            return path
        plt.show()
        return None

    # ------------------------------------------------------------------
    # Fallback plain-text output
    # ------------------------------------------------------------------

    def _plain_bb84_summary(self, result) -> None:
        print(f"\n{'='*50}")
        print(f"BB84 Session [{result.session_id}]")
        print(f"  QBER: {result.qber:.2%}  Secure: {result.secure}")
        print(f"  Eve detected: {result.eve_detected}")

    def _plain_grover_result(self, result) -> None:
        print(f"\n{'='*50}")
        print(f"Grover Search: N={result.search_space_size}")
        print(f"  Found: {result.found}  Measured: {result.measured_state}")
        print(f"  Speedup: {result.speedup_factor:.1f}x")
