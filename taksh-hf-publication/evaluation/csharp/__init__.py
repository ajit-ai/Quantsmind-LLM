"""C# and .NET evaluation support for Taksh."""

from evaluation.csharp.diagnostics import (
    CompilerDiagnostic,
    analyse_csharp_source,
    parse_roslyn_diagnostics,
    summarize_diagnostics,
)
from evaluation.csharp.runner import (
    CSharpEnvironment,
    CSharpRunResult,
    detect_csharp_environment,
    compile_csharp,
)
from evaluation.csharp.suite import CSHARP_PROBES, probes_by_category

__all__ = [
    "CompilerDiagnostic",
    "CSharpEnvironment",
    "CSharpRunResult",
    "CSHARP_PROBES",
    "analyse_csharp_source",
    "compile_csharp",
    "detect_csharp_environment",
    "parse_roslyn_diagnostics",
    "probes_by_category",
    "summarize_diagnostics",
]
