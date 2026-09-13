"""
Helper utilities for SmartBank AI.

Contains formatting functions, color maps, and common utilities
used across modules and UI components.
"""

from __future__ import annotations

from datetime import datetime


# ─── Currency & Number Formatting ────────────────────────────────────────────

def format_currency(amount: float) -> str:
    """Format amount as Indian Rupees with commas.

    Args:
        amount: Numeric amount.

    Returns:
        Formatted string like '₹12,50,000'.
    """
    if amount < 0:
        return f"-₹{format_indian_number(abs(amount))}"
    return f"₹{format_indian_number(amount)}"


def format_indian_number(num: float) -> str:
    """Format number in Indian numbering system (lakhs, crores).

    Args:
        num: Number to format.

    Returns:
        Formatted string with Indian comma placement.
    """
    if num < 1000:
        return f"{num:,.2f}" if num != int(num) else f"{int(num)}"

    s = str(int(num))
    last_three = s[-3:]
    rest = s[:-3]

    # Add commas every 2 digits in the remaining part
    parts = []
    while rest:
        parts.append(rest[-2:])
        rest = rest[:-2]

    return ",".join(reversed(parts)) + "," + last_three


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a decimal value as a percentage string.

    Args:
        value: Decimal value (e.g., 0.42 for 42%).
        decimals: Number of decimal places.

    Returns:
        Formatted string like '42.0%'.
    """
    return f"{value * 100:.{decimals}f}%"


# ─── Risk Level Colors ──────────────────────────────────────────────────────

RISK_COLORS = {
    "LOW": "#28a745",
    "MEDIUM": "#ffc107",
    "HIGH": "#dc3545",
}

DECISION_COLORS = {
    "APPROVE": "#28a745",
    "REVIEW": "#ffc107",
    "REJECT": "#dc3545",
}

ACTION_COLORS = {
    "Allow": "#28a745",
    "Monitor": "#17a2b8",
    "Additional Verification": "#ffc107",
    "Block": "#dc3545",
}


def get_risk_color(level: str) -> str:
    """Get hex color for a risk level.

    Args:
        level: Risk level string (LOW, MEDIUM, HIGH).

    Returns:
        Hex color string.
    """
    return RISK_COLORS.get(level.upper(), "#6c757d")


def get_risk_emoji(level: str) -> str:
    """Get emoji indicator for a risk level.

    Args:
        level: Risk level string.

    Returns:
        Emoji string.
    """
    emojis = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
    return emojis.get(level.upper(), "⚪")


def get_decision_emoji(decision: str) -> str:
    """Get emoji for a loan decision.

    Args:
        decision: Decision string (APPROVE, REVIEW, REJECT).

    Returns:
        Emoji string.
    """
    emojis = {"APPROVE": "✅", "REVIEW": "⚠️", "REJECT": "❌"}
    return emojis.get(decision.upper(), "❓")


# ─── Timestamp Helpers ───────────────────────────────────────────────────────

def now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat()


def format_timestamp(iso_str: str) -> str:
    """Format an ISO timestamp for display.

    Args:
        iso_str: ISO format timestamp string.

    Returns:
        Human-readable date-time string.
    """
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%d %b %Y, %H:%M")
    except (ValueError, TypeError):
        return str(iso_str)
