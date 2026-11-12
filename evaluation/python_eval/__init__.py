"""Python coding capability evaluation framework (Taksh v1.4.0).

The framework is language-neutral in its *schema* and *dispatch*: a case
carries a language identifier, and validators are registered per language.
Python is the only language with cases in this version; the schema is written
so that a later language needs a new case file and a validator, not a new
engine.

Modules
-------
``schema``    Versioned case schema, validation, contamination classification
``syntax``    Language dispatch and per-language syntax validators
``sandbox``   Isolation capability detection and the guarded executor
``metrics``   Aggregation into the reported metric set
``runner``    Case-by-case execution and reporting
"""

SCHEMA_VERSION = "quantsmind.eval_case/1"
REPORT_SCHEMA_VERSION = "quantsmind.eval_report/1"
PROFILE_SCHEMA_VERSION = "quantsmind.agent_profile/1"

#: Categories the Python capability suite covers.
CATEGORIES = (
    "syntax_validity",
    "function_completion",
    "small_algorithm",
    "string_and_list",
    "control_flow",
    "debugging",
    "test_generation",
    "format_compliance",
)

#: How a case relates to any training data. Authored in this repository and
#: not drawn from a published suite, so it is ``authored``; the field exists so
#: a future case file can declare ``published_suite`` or ``seen_in_training``.
CONTAMINATION_CLASSES = ("authored", "published_suite", "seen_in_training", "unknown")

#: Execution environment a case expects when it is executed functionally.
EXECUTION_ENVIRONMENTS = ("none", "python3", "python3-isolated")
