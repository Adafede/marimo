"""Element and halogen configuration for molecular formula filters."""

__all__ = [
    "ELEMENT_CONFIGS",
    "HALOGEN_CONFIGS",
    "ELEMENT_DEFAULTS",
]

# Element configurations: (symbol, name, max_count)
ELEMENT_CONFIGS = [
    ("C", "carbon", 100),
    ("H", "hydrogen", 200),
    ("N", "nitrogen", 50),
    ("O", "oxygen", 50),
    ("P", "phosphorus", 20),
    ("S", "sulfur", 20),
]

# Halogen configurations: (symbol, name)
HALOGEN_CONFIGS = [
    ("F", "fluorine"),
    ("Cl", "chlorine"),
    ("Br", "bromine"),
    ("I", "iodine"),
]

# Default max values by element symbol
ELEMENT_DEFAULTS = {elem[0].lower(): elem[2] for elem in ELEMENT_CONFIGS}
