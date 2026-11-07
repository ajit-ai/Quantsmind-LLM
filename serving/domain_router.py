# Taksh v2.2.0 — Domain Model Router
#
# Routes requests to the appropriate domain-specific model based on
# the instruction content. Falls back to the general v2.0.0 model
# when no domain is confidently detected.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RoutingDecision:
    domain: str
    confidence: float
    model_name: str
    reason: str


class DomainRouter:
    """Routes coding tasks to domain-specific models.

    Uses keyword matching and heuristics to determine which domain-specific
    model should handle a request. Falls back to the general model when
    confidence is below threshold or no domain matches.
    """

    DOMAIN_KEYWORDS = {
        "frontend": {
            "react", "typescript", "jsx", "tsx", "css", "html", "component",
            "component", "ui", "interface", "render", "state", "hook",
            "useState", "useEffect", "props", "stylesheet", "dom", "css",
            "button", "modal", "form", "navbar", "sidebar", "accessibility",
            "aria", "keyboard", "semantic", "responsive", "flexbox", "grid",
            "media query", "bootstrap", "tailwind", "material-ui",
        },
        "backend": {
            "api", "rest", "graphql", "endpoint", "route", "controller",
            "middleware", "database", "sql", "postgresql", "mysql", "mongodb",
            "redis", "auth", "jwt", "oauth", "session", "middleware",
            "deployment", "ci/cd", "docker", "kubernetes", "aws", "lambda",
            "server", "endpoint", "request", "response", "status code",
            "pagination", "filtering", "sorting", "migration", "schema",
        },
        "datascience": {
            "pandas", "numpy", "matplotlib", "seaborn", "plotly", "sklearn",
            "tensorflow", "pytorch", "keras", "dataframe", "visualization",
            "eda", "exploratory", "model", "train", "predict", "regression",
            "classification", "clustering", "sql", "query", "dataset",
            "feature", "preprocessing", "normalization", "encoding",
            "cross-validation", "metric", "accuracy", "precision", "recall",
        },
        "embedded": {
            "c", "c++", "arduino", "risc", "arm", "avr", "esp32", "stm32",
            "gpio", "i2c", "spi", "uart", "rtos", "freertos", "interrupt",
            "dma", "register", "memory", "heap", "stack", "bare metal",
            "bootloader", "watchdog", "timer", "adc", "pwm", "dac",
            "microcontroller", "embedded", "firmware", "bare-metal",
        },
    }

    def __init__(self, general_model_name: str = "qm-code-v200-7b"):
        self.general_model_name = general_model_name
        self.domain_models = {
            "frontend": "qm-code-v220-frontend-7b",
            "backend": "qm-code-v220-backend-7b",
            "datascience": "qm-code-v220-datascience-7b",
            "embedded": "qm-code-v220-embedded-7b",
        }

    def route(self, prompt: str, context: dict | None = None) -> RoutingDecision:
        """Route a request to the appropriate domain model.

        Args:
            prompt: The user's coding instruction.
            context: Optional context (file extensions, imports, etc.).

        Returns:
            RoutingDecision with domain, confidence, model name, and reason.
        """
        prompt_lower = prompt.lower()

        # Score each domain based on keyword matches
        domain_scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in prompt_lower)
            if score > 0:
                domain_scores[domain] = score

        if not domain_scores:
            return RoutingDecision(
                domain="general",
                confidence=0.0,
                model_name=self.general_model_name,
                reason="No domain keywords matched; using general model",
            )

        # Find the best matching domain
        best_domain = max(domain_scores, key=domain_scores.get)
        total_score = sum(domain_scores.values())
        confidence = domain_scores[best_domain] / total_score

        # If confidence is low, fall back to general model
        if confidence < 0.4:
            return RoutingDecision(
                domain="general",
                confidence=confidence,
                model_name=self.general_model_name,
                reason=f"Low confidence ({confidence:.2f}) for {best_domain}; using general model",
            )

        model_name = self.domain_models[best_domain]
        return RoutingDecision(
            domain=best_domain,
            confidence=confidence,
            model_name=model_name,
            reason=f"Matched {domain_scores[best_domain]} keyword(s) for {best_domain}",
        )

    def route_with_context(
        self,
        prompt: str,
        context: dict | None = None,
    ) -> RoutingDecision:
        """Route with additional context (file extensions, imports, etc.)."""
        base = self.route(prompt, context)

        if context:
            # Boost confidence based on file extensions
            file_exts = context.get("file_extensions", [])
            import_statements = context.get("import_statements", [])

            ext_to_domain = {
                ".tsx": "frontend",
                ".ts": "frontend",
                ".jsx": "frontend",
                ".css": "frontend",
                ".html": "frontend",
                ".py": "backend",  # Could be backend or datascience
                ".java": "backend",
                ".go": "backend",
                ".rs": "backend",
                ".c": "embedded",
                ".cpp": "embedded",
                ".h": "embedded",
                ".ino": "embedded",
            }

            for ext in file_exts:
                if ext in ext_to_domain:
                    domain = ext_to_domain[ext]
                    if domain == base.domain:
                        base.confidence = min(1.0, base.confidence + 0.2)
                        base.reason += f"; file extension {ext} confirms {domain}"
                        break

            # Check imports
            domain_imports = {
                "react": "frontend",
                "django": "backend",
                "flask": "backend",
                "pandas": "datascience",
                "torch": "datascience",
                "sklearn": "datascience",
                "arduino": "embedded",
                "freertos": "embedded",
                "esp32": "embedded",
            }

            for imp in import_statements:
                imp_lower = imp.lower()
                for keyword, domain in domain_imports.items():
                    if keyword in imp_lower:
                        if domain == base.domain:
                            base.confidence = min(1.0, base.confidence + 0.15)
                            base.reason += f"; import {imp} confirms {domain}"
                            break

        return base


# Global router instance
router = DomainRouter()


def route_request(prompt: str, context: dict | None = None) -> RoutingDecision:
    """Route a coding request to the appropriate domain model.

    Convenience function that uses the global router instance.

    Args:
        prompt: The user's coding instruction.
        context: Optional context (file extensions, imports, etc.).

    Returns:
        RoutingDecision with domain, confidence, model name, and reason.
    """
    return router.route_with_context(prompt, context)
