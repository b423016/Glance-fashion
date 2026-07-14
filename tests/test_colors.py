"""CIELab colour naming + query→region colour gate."""
from fashionlib import colors


def test_red_gate_accepts_maroon_rejects_blue():
    maroon = colors._lab((110, 20, 30))
    blue = colors._lab((40, 80, 205))
    assert colors.color_gate(maroon, False, "red") > 0.5   # same family
    assert colors.color_gate(blue, False, "red") < 0.3     # different hue


def test_white_gate_requires_achromatic():
    red = colors._lab((200, 30, 30))
    white = colors._lab((245, 245, 245))
    assert colors.color_gate(red, False, "white") < 0.2    # a red garment isn't "white"
    assert colors.color_gate(white, True, "white") > 0.6


def test_unknown_colour_is_neutral():
    assert colors.color_gate(colors._lab((40, 80, 205)), False, "turquoise") == 0.5


def test_name_from_lab():
    assert colors.name_from_lab(colors._lab((10, 10, 10))) == "black"
    assert colors.name_from_lab(colors._lab((245, 245, 245))) == "white"
    assert colors.name_from_lab(colors._lab((40, 80, 205))) == "blue"


def test_family_folds_synonyms():
    assert colors.family("maroon") == "red"
    assert colors.family("navy") == "blue"
    assert colors.family("green") == "green"
