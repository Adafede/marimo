"""Element and halogen configuration for molecular formula filters."""

__all__ = [
    "ELEMENT_CONFIGS",
    "HALOGEN_CONFIGS",
    "ELEMENT_DEFAULTS",
]

# Element configurations: (symbol, name, min_count, max_count)
ELEMENT_CONFIGS = [
    ("C", "carbon", 0, 100),
    ("H", "hydrogen", 0, 200),
    ("N", "nitrogen", 0, 50),
    ("O", "oxygen", 0, 50),
    ("P", "phosphorus", 0, 20),
    ("S", "sulfur", 0, 20),
]

# Halogen configurations: (symbol, name)
HALOGEN_CONFIGS = [
    ("F", "fluorine"),
    ("Cl", "chlorine"),
    ("Br", "bromine"),
    ("I", "iodine"),
]

# Default max values by element symbol
ELEMENT_DEFAULTS = {elem[0].lower(): elem[3] for elem in ELEMENT_CONFIGS}
