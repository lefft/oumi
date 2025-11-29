# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional

import pandas as pd
import typer
from rich.table import Table

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.utils.logging import logger

# Valid output formats for analysis results
_VALID_OUTPUT_FORMATS = ("csv", "json", "parquet")

if TYPE_CHECKING:
    from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer


def analyze(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path to the configuration file for analysis.",
        ),
    ],
    output: Annotated[
        Optional[str],
        typer.Option(
            "--output",
            "-o",
            help="Output directory for analysis results. Overrides config output_path.",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format for results: csv, json, or parquet (case-insensitive).",
        ),
    ] = "csv",
    level: cli_utils.LOG_LEVEL_TYPE = None,
    verbose: cli_utils.VERBOSE_TYPE = False,
):
    """Analyze a dataset to compute metrics and statistics.

    This command loads a dataset and applies configured analyzers to compute
    metrics like text length, token counts, and other statistics. Results can
    be exported to various formats for further analysis.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for analysis.
        output: Output directory for results. Overrides config output_path.
            Must be a writable directory path. Will be created if it doesn't exist.
        output_format: Output format (csv, json, parquet). Case-insensitive.
        level: The logging level for the specified command.
        verbose: Enable verbose logging with additional debug information.

    Raises:
        typer.Exit: If an error occurs during analysis or export.
    """
    # Import DatasetAnalyzer type for type hints
    from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer

    # Validate output format early before any expensive operations
    output_format = output_format.lower()
    if output_format not in _VALID_OUTPUT_FORMATS:
        cli_utils.CONSOLE.print(
            f"[red]Error:[/red] Invalid output format '{output_format}'. "
            f"Supported formats: {', '.join(_VALID_OUTPUT_FORMATS)}"
        )
        raise typer.Exit(code=1)

    try:
        extra_args = cli_utils.parse_extra_cli_args(ctx)

        config = str(
            cli_utils.resolve_and_fetch_config(
                try_get_config_name_for_alias(config, AliasType.ANALYZE),
            )
        )

        with cli_utils.CONSOLE.status(
            "[green]Loading configuration...[/green]", spinner="dots"
        ):
            # Delayed imports
            from oumi.core.configs import AnalyzeConfig

        # Load configuration
        parsed_config: AnalyzeConfig = AnalyzeConfig.from_yaml_and_arg_list(
            config, extra_args, logger=logger
        )

        # Override output path if provided via CLI
        if output:
            parsed_config.output_path = output

        # Validate configuration
        parsed_config.finalize_and_validate()

        if verbose:
            parsed_config.print_config(logger)

        # Create analyzer
        with cli_utils.CONSOLE.status(
            "[green]Loading dataset...[/green]", spinner="dots"
        ):
            analyzer = DatasetAnalyzer(parsed_config)

        # Run analysis
        with cli_utils.CONSOLE.status(
            "[green]Running analysis...[/green]", spinner="dots"
        ):
            analyzer.analyze_dataset()

        # Display summary
        _display_analysis_summary(analyzer)

        # Export results
        if parsed_config.output_path:
            _export_results(analyzer, parsed_config.output_path, output_format)

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        cli_utils.CONSOLE.print(
            f"[red]Error:[/red] Configuration file not found: {e}\n"
            f"Please check the path and try again."
        )
        raise typer.Exit(code=1)

    except ValueError as e:
        logger.error(f"Invalid configuration or parameters: {e}")
        cli_utils.CONSOLE.print(
            f"[red]Error:[/red] Invalid configuration: {e}\n"
            f"Please check your configuration file and parameters."
        )
        raise typer.Exit(code=1)

    except RuntimeError as e:
        logger.error(f"Analysis failed: {e}")
        cli_utils.CONSOLE.print(f"[red]Error:[/red] Analysis failed: {e}")
        raise typer.Exit(code=1)

    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
        cli_utils.CONSOLE.print(
            f"[red]Unexpected error:[/red] {e}\nPlease check the logs for more details."
        )
        raise typer.Exit(code=1)


def _display_analysis_summary(analyzer: "DatasetAnalyzer") -> None:
    """Display analysis summary in formatted tables to the console.

    Presents dataset overview, message-level metrics, and conversation
    turn statistics in rich-formatted tables using the console.

    Args:
        analyzer: The dataset analyzer containing analysis results.
    """
    summary = analyzer.analysis_summary

    # Dataset overview table
    overview = summary.get("dataset_overview", {})
    if overview:
        table = Table(
            title="Dataset Overview",
            title_style="bold magenta",
            show_lines=True,
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Dataset Name", str(overview.get("dataset_name", "N/A")))
        table.add_row(
            "Total Conversations", str(overview.get("total_conversations", "N/A"))
        )
        table.add_row(
            "Conversations Analyzed", str(overview.get("conversations_analyzed", "N/A"))
        )
        table.add_row(
            "Coverage",
            f"{overview.get('dataset_coverage_percentage', 0):.1f}%",
        )
        table.add_row("Total Messages", str(overview.get("total_messages", "N/A")))
        table.add_row(
            "Analyzers Used",
            ", ".join(overview.get("analyzers_used", [])) or "None",
        )
        cli_utils.CONSOLE.print(table)

    # Message-level summary
    msg_summary = summary.get("message_level_summary", {})
    if msg_summary:
        for analyzer_name, metrics in msg_summary.items():
            table = Table(
                title=f"Message-Level Metrics ({analyzer_name})",
                title_style="bold blue",
                show_lines=True,
            )
            table.add_column("Metric", style="cyan")
            table.add_column("Mean", style="green")
            table.add_column("Std", style="yellow")
            table.add_column("Min", style="dim")
            table.add_column("Max", style="dim")
            table.add_column("Median", style="dim")

            for metric_name, stats in metrics.items():
                if isinstance(stats, dict):
                    table.add_row(
                        metric_name,
                        f"{stats.get('mean', 'N/A'):.2f}"
                        if isinstance(stats.get("mean"), (int, float))
                        else "N/A",
                        f"{stats.get('std', 'N/A'):.2f}"
                        if isinstance(stats.get("std"), (int, float))
                        else "N/A",
                        str(stats.get("min", "N/A")),
                        str(stats.get("max", "N/A")),
                        f"{stats.get('median', 'N/A'):.2f}"
                        if isinstance(stats.get("median"), (int, float))
                        else "N/A",
                    )
            cli_utils.CONSOLE.print(table)

    # Conversation turns summary
    turns_summary = summary.get("conversation_turns", {})
    if turns_summary and isinstance(turns_summary, dict) and turns_summary.get("count"):
        table = Table(
            title="Conversation Turns",
            title_style="bold yellow",
            show_lines=True,
        )
        table.add_column("Statistic", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Count", str(turns_summary.get("count", "N/A")))
        table.add_row(
            "Mean",
            f"{turns_summary.get('mean', 0):.2f}"
            if isinstance(turns_summary.get("mean"), (int, float))
            else "N/A",
        )
        table.add_row(
            "Std",
            f"{turns_summary.get('std', 0):.2f}"
            if isinstance(turns_summary.get("std"), (int, float))
            else "N/A",
        )
        table.add_row("Min", str(turns_summary.get("min", "N/A")))
        table.add_row("Max", str(turns_summary.get("max", "N/A")))
        table.add_row(
            "Median",
            f"{turns_summary.get('median', 0):.2f}"
            if isinstance(turns_summary.get("median"), (int, float))
            else "N/A",
        )
        cli_utils.CONSOLE.print(table)


def _export_results(
    analyzer: "DatasetAnalyzer",  # noqa: F821
    output_path: str,
    output_format: str,
) -> None:
    """Export analysis results to files.

    Args:
        analyzer: The dataset analyzer containing analysis results.
        output_path: Directory path where results will be saved.
        output_format: Output format ('csv', 'json', or 'parquet').

    Raises:
        RuntimeError: If directory creation or file export fails.
    """
    # Create output directory with error handling
    try:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        logger.error(f"Permission denied: Cannot create directory '{output_path}'")
        raise RuntimeError(
            f"Cannot create output directory '{output_path}'. "
            f"Check file permissions and try again."
        )
    except OSError as e:
        logger.error(f"Failed to create output directory '{output_path}': {e}")
        raise RuntimeError(
            f"Cannot create output directory '{output_path}': {e}. "
            f"Check disk space and path validity."
        )

    # Export message-level results
    if analyzer.message_df is not None and not analyzer.message_df.empty:
        msg_path = output_dir / f"message_analysis.{output_format}"
        try:
            _save_dataframe(analyzer.message_df, msg_path, output_format)
            cli_utils.CONSOLE.print(
                f"[green]Saved message analysis to:[/green] {msg_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save message analysis to '{msg_path}': {e}")
            raise RuntimeError(
                f"Failed to export message analysis: {e}. "
                f"Check disk space and file permissions."
            )

    # Export conversation-level results
    if analyzer.conversation_df is not None and not analyzer.conversation_df.empty:
        conv_path = output_dir / f"conversation_analysis.{output_format}"
        try:
            _save_dataframe(analyzer.conversation_df, conv_path, output_format)
            cli_utils.CONSOLE.print(
                f"[green]Saved conversation analysis to:[/green] {conv_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save conversation analysis to '{conv_path}': {e}")
            raise RuntimeError(
                f"Failed to export conversation analysis: {e}. "
                f"Check disk space and file permissions."
            )

    # Export summary as JSON
    summary_path = output_dir / "analysis_summary.json"
    try:
        with open(summary_path, "w") as f:
            json.dump(analyzer.analysis_summary, f, indent=2, default=str)
        cli_utils.CONSOLE.print(
            f"[green]Saved analysis summary to:[/green] {summary_path}"
        )
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to save analysis summary to '{summary_path}': {e}")
        raise RuntimeError(
            f"Failed to export analysis summary: {e}. "
            f"Check disk space and file permissions."
        )
    except TypeError as e:
        logger.error(f"Failed to serialize analysis summary: {e}")
        raise RuntimeError(
            f"Failed to serialize analysis summary to JSON: {e}. "
            f"Some data may not be JSON-serializable."
        )


def _save_dataframe(df: pd.DataFrame, path: Path, output_format: str) -> None:
    """Save a DataFrame to the specified format.

    Args:
        df: DataFrame to save.
        path: Output file path.
        output_format: Output format ('csv', 'json', or 'parquet').

    Raises:
        OSError: If file write fails.
        ValueError: If data cannot be serialized.
    """
    try:
        if output_format == "csv":
            df.to_csv(path, index=False)
        elif output_format == "json":
            df.to_json(path, orient="records", indent=2)
        elif output_format == "parquet":
            df.to_parquet(path, index=False)
    except (OSError, PermissionError) as e:
        raise OSError(f"Cannot write to file '{path}': {e}")
    except ValueError as e:
        raise ValueError(f"Cannot serialize DataFrame to {output_format}: {e}")
