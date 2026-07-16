"""Fashion domain lexicons: canonical colors, garment types, scenes, styles.

This is the shared vocabulary that makes structural attribute binding work:
the VLM extractor canonicalizes INTO it, and the query parser expands OUT of it,
so 'crimson blouse' at index time meets 'red shirt' at query time on exact terms.
Pure data + small pure functions. No ML imports.
"""
from __future__ import annotations

# ── colors ───────────────────────────────────────────────────────────────────
# canonical color -> representative hex anchor (for nearest-color mapping)
CANONICAL_COLORS: dict[str, str] = {
    "black": "#1a1a1a", "white": "#f5f5f5", "gray": "#808080",
    "red": "#d0312d", "orange": "#f28c28", "yellow": "#ffd400",
    "green": "#3a7d44", "blue": "#2a6bd4", "navy": "#1b2a4a",
    "purple": "#7d3cb5", "pink": "#e75480", "brown": "#7b4a2d",
    "beige": "#d9c7a7", "cream": "#f4ead5", "gold": "#c9a227",
    "silver": "#c0c0c0", "teal": "#2a9d9f", "maroon": "#701c2c",
    "olive": "#6b7233", "khaki": "#b7a878",
}

COLOR_ALIASES: dict[str, str] = {
    "crimson": "red", "scarlet": "red", "cherry": "red", "wine": "maroon",
    "burgundy": "maroon", "rust": "orange", "coral": "orange", "peach": "orange",
    "mustard": "yellow", "lemon": "yellow", "golden": "gold", "amber": "yellow",
    "lime": "green", "emerald": "green", "forest green": "green", "mint": "green",
    "sky blue": "blue", "light blue": "blue", "royal blue": "blue", "cobalt": "blue",
    "azure": "blue", "denim blue": "blue", "dark blue": "navy", "midnight blue": "navy",
    "turquoise": "teal", "cyan": "teal", "aqua": "teal",
    "violet": "purple", "lavender": "purple", "lilac": "purple", "magenta": "pink",
    "fuchsia": "pink", "rose": "pink", "salmon": "pink", "blush": "pink",
    "tan": "beige", "camel": "beige", "sand": "beige", "taupe": "beige",
    "nude": "beige", "ivory": "cream", "off-white": "cream", "off white": "cream",
    "charcoal": "gray", "grey": "gray", "slate": "gray", "gunmetal": "gray",
    "chocolate": "brown", "coffee": "brown", "chestnut": "brown", "mahogany": "brown",
    "bright yellow": "yellow", "dark green": "green", "light gray": "gray",
    "light grey": "gray", "dark gray": "gray", "dark grey": "gray",
    "multicolor": "multicolor", "multicolour": "multicolor", "multi": "multicolor",
    "patterned": "multicolor", "plaid": "multicolor", "striped": "multicolor",
    "floral": "multicolor", "print": "multicolor",
}


def canonical_color(name: str | None, hex_code: str | None = None) -> str | None:
    """Map an arbitrary color mention to the canonical palette.

    Order: exact canonical -> alias table -> substring alias ('bright red' -> 'red')
    -> nearest anchor by RGB distance if a hex is available -> None.
    """
    if name:
        n = " ".join(name.strip().lower().split())
        if n in CANONICAL_COLORS or n == "multicolor":
            return n
        if n in COLOR_ALIASES:
            return COLOR_ALIASES[n]
        for token in (n.split()[-1],):  # 'bright yellow' -> 'yellow'
            if token in CANONICAL_COLORS:
                return token
            if token in COLOR_ALIASES:
                return COLOR_ALIASES[token]
    if hex_code:
        return nearest_color(hex_code)
    return None


def nearest_color(hex_code: str) -> str | None:
    h = hex_code.strip().lstrip("#")
    if len(h) != 6:
        return None
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return None
    best, best_d = None, float("inf")
    for name, anchor in CANONICAL_COLORS.items():
        a = anchor.lstrip("#")
        ar, ag, ab = int(a[0:2], 16), int(a[2:4], 16), int(a[4:6], 16)
        d = (r - ar) ** 2 + (g - ag) ** 2 + (b - ab) ** 2
        if d < best_d:
            best, best_d = name, d
    return best


# ── garment types ────────────────────────────────────────────────────────────
# canonical type -> aliases (query-language + VLM-language). Canonical names
# align with Fashionpedia supercategories so detector labels join cleanly.
GARMENT_TYPES: dict[str, list[str]] = {
    "shirt":      ["shirt", "blouse", "button-down", "button down", "button-up",
                   "dress shirt", "flannel", "polo", "polo shirt"],
    "t-shirt":    ["t-shirt", "tshirt", "tee", "t shirt", "top", "tank top",
                   "crop top", "camisole"],
    "sweater":    ["sweater", "pullover", "jumper", "knit", "turtleneck"],
    "sweatshirt": ["sweatshirt", "hoodie", "hooded sweatshirt"],
    "cardigan":   ["cardigan"],
    "jacket":     ["jacket", "blazer", "sport coat", "sports jacket", "suit jacket",
                   "suit", "bomber", "denim jacket", "leather jacket", "windbreaker",
                   "puffer"],
    "coat":       ["coat", "raincoat", "rain coat", "trench coat", "trench",
                   "overcoat", "parka", "anorak", "peacoat"],
    "vest":       ["vest", "waistcoat", "gilet"],
    "dress":      ["dress", "gown", "sundress", "maxi dress"],
    "jumpsuit":   ["jumpsuit", "romper", "overalls", "dungarees"],
    "pants":      ["pants", "trousers", "jeans", "chinos", "slacks", "joggers",
                   "sweatpants", "leggings", "cargo pants"],
    "shorts":     ["shorts"],
    "skirt":      ["skirt", "miniskirt", "midi skirt"],
    "tie":        ["tie", "necktie", "bow tie", "bowtie"],
    "hat":        ["hat", "cap", "beanie", "baseball cap", "fedora", "sun hat"],
    "scarf":      ["scarf"],
    "glove":      ["glove", "gloves", "mittens"],
    "belt":       ["belt"],
    "glasses":    ["glasses", "sunglasses", "eyeglasses", "shades"],
    "watch":      ["watch", "wristwatch"],
    "shoe":       ["shoe", "shoes", "sneakers", "trainers", "boots", "heels",
                   "loafers", "sandals", "oxfords", "flats"],
    "sock":       ["sock", "socks"],
    "tights":     ["tights", "stockings", "pantyhose"],
    "bag":        ["bag", "handbag", "purse", "backpack", "tote", "wallet",
                   "satchel", "messenger bag"],
    "umbrella":   ["umbrella"],
}

# reverse index: alias -> canonical
_ALIAS_TO_TYPE: dict[str, str] = {
    alias: canon for canon, aliases in GARMENT_TYPES.items() for alias in aliases
}

# Fashionpedia detector label -> canonical type (detector emits these exact strings)
FASHIONPEDIA_LABEL_MAP: dict[str, str] = {
    "shirt, blouse": "shirt",
    "top, t-shirt, sweatshirt": "t-shirt",
    "sweater": "sweater",
    "cardigan": "cardigan",
    "jacket": "jacket",
    "vest": "vest",
    "pants": "pants",
    "shorts": "shorts",
    "skirt": "skirt",
    "coat": "coat",
    "dress": "dress",
    "jumpsuit": "jumpsuit",
    "cape": "coat",
    "glasses": "glasses",
    "hat": "hat",
    "headband, head covering, hair accessory": "hat",
    "tie": "tie",
    "glove": "glove",
    "watch": "watch",
    "belt": "belt",
    "leg warmer": "sock",
    "tights, stockings": "tights",
    "sock": "sock",
    "shoe": "shoe",
    "bag, wallet": "bag",
    "scarf": "scarf",
    "umbrella": "umbrella",
}


def canonical_garment(term: str | None) -> str | None:
    if not term:
        return None
    t = " ".join(term.strip().lower().split())
    if t in GARMENT_TYPES:
        return t
    if t in _ALIAS_TO_TYPE:
        return _ALIAS_TO_TYPE[t]
    if t in FASHIONPEDIA_LABEL_MAP:
        return FASHIONPEDIA_LABEL_MAP[t]
    if t.endswith("s") and t[:-1] in _ALIAS_TO_TYPE:      # crude plural
        return _ALIAS_TO_TYPE[t[:-1]]
    return None


# ── materials (negation support: "no denim") ────────────────────────────────
MATERIALS: dict[str, list[str]] = {
    "denim":   ["denim", "jean", "jeans"],
    "leather": ["leather", "suede"],
    "wool":    ["wool", "woolen", "knit", "cashmere"],
    "silk":    ["silk", "satin"],
    "cotton":  ["cotton"],
    "linen":   ["linen"],
    "fur":     ["fur", "faux fur"],
}
_ALIAS_TO_MATERIAL: dict[str, str] = {
    a: canon for canon, aliases in MATERIALS.items() for a in aliases
}


def canonical_material(term: str | None) -> str | None:
    if not term:
        return None
    t = " ".join(term.strip().lower().split())
    return _ALIAS_TO_MATERIAL.get(t)


# ── scenes ───────────────────────────────────────────────────────────────────
SCENE_LEXICON: dict[str, list[str]] = {
    "office":     ["office", "workplace", "meeting room", "conference room",
                   "boardroom", "cubicle", "desk", "corporate"],
    "street":     ["street", "city", "urban", "sidewalk", "downtown", "crosswalk",
                   "city walk", "road", "alley", "outside a building"],
    "park":       ["park", "garden", "bench", "park bench", "lawn", "trees",
                   "greenery", "meadow", "picnic"],
    "home":       ["home", "living room", "bedroom", "kitchen", "couch", "sofa",
                   "apartment", "indoors at home"],
    "beach":      ["beach", "seaside", "ocean", "sand", "coast"],
    "restaurant": ["restaurant", "cafe", "coffee shop", "bar", "diner"],
    "studio":     ["studio", "plain background", "white background", "runway",
                   "catwalk", "photoshoot"],
}
_ALIAS_TO_SCENE: dict[str, str] = {
    a: canon for canon, aliases in SCENE_LEXICON.items() for a in aliases
}


def canonical_scene(term: str | None) -> str | None:
    if not term:
        return None
    t = " ".join(term.strip().lower().split())
    if t in SCENE_LEXICON:
        return t
    if t in _ALIAS_TO_SCENE:
        return _ALIAS_TO_SCENE[t]
    for alias, canon in _ALIAS_TO_SCENE.items():   # substring: "modern office interior"
        if alias in t:
            return canon
    return None


# ── style / formality ────────────────────────────────────────────────────────
STYLE_LEXICON: dict[str, list[str]] = {
    "formal":       ["formal", "black tie", "elegant", "dressy"],
    "business":     ["business", "professional", "office attire", "business attire",
                     "corporate", "work attire", "suit and tie"],
    "smart-casual": ["smart casual", "smart-casual", "business casual"],
    "casual":       ["casual", "weekend", "relaxed", "everyday", "laid back",
                     "laid-back", "comfortable", "streetwear"],
    "sport":        ["sport", "sporty", "athletic", "gym", "workout", "athleisure",
                     "running"],
}
_ALIAS_TO_STYLE: dict[str, str] = {
    a: canon for canon, aliases in STYLE_LEXICON.items() for a in aliases
}


def canonical_style(term: str | None) -> str | None:
    if not term:
        return None
    t = " ".join(term.strip().lower().split())
    if t in STYLE_LEXICON:
        return t
    if t in _ALIAS_TO_STYLE:
        return _ALIAS_TO_STYLE[t]
    for alias, canon in _ALIAS_TO_STYLE.items():
        if alias in t:
            return canon
    return None


# formality compatibility groups used by the attribute scorer
STYLE_TO_FORMALITY: dict[str, set[str]] = {
    "formal":       {"formal", "business"},
    "business":     {"business", "formal", "smart-casual"},
    "smart-casual": {"smart-casual", "business", "casual"},
    "casual":       {"casual", "smart-casual", "sport"},
    "sport":        {"sport", "casual"},
}
