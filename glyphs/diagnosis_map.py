"""Taxonomy -> diagnosis groups.

The CSV column 'Associated diagnoses (as listed in source)' says in which diagnoses an IDS structure occurs; it carries no
sensitivity / specificity. We normalise the free text into 8 groups (ISIC-style) and spread each structure's weight evenly over
the groups it lists (a structure listed for 1 group is more specific than one listed for 4). Generic entries ('Various diagnoses')
give no evidence. This is a transparent heuristic, not a trained or validated diagnostic model.
"""
import csv, re

GROUPS = {"MEL": "Melanom", "NV": "Melanozytärer Nävus", "BCC": "Basalzellkarzinom", "AKIEC": "Aktinische Keratose / SCC / Bowen",
          "BKL": "Benigne Keratose / Lentigo", "DF": "Dermatofibrom", "VASC": "Vaskuläre Läsion", "OTHER": "Sonstige"}
# token (lower case) -> (display name, groups)
TOK = [
    (r"^(flat )?bcc$", "BCC", ["BCC"]),
    (r"^melanoma$", "Melanom", ["MEL"]), (r"^acral melanoma$", "Akrales Melanom", ["MEL"]), (r"^lentiginous melanomas$", "Lentiginöses Melanom", ["MEL"]),
    (r"^lentigo maligna$", "Lentigo maligna", ["MEL"]), (r"^(melanoma )?metastasis$", "Metastase", ["MEL"]),
    (r"^seborrheic keratosis$", "Seborrhoische Keratose", ["BKL"]), (r"^solar lentigo$", "Lentigo solaris", ["BKL"]),
    (r"^lichen planus-like keratosis$", "Lichenoide Keratose", ["BKL"]),
    (r"^acral nevi$", "Akrale Nävi", ["NV"]), (r"^dermal nevi$", "Dermale Nävi", ["NV"]), (r"^melanocytic nevi$", "Melanozytäre Nävi", ["NV"]),
    (r"^reed nevus$", "Reed-Nävus", ["NV"]), (r"^spitz nevus$", "Spitz-Nävus", ["NV"]), (r"^recurrent nevi$|^recurrent nevus$", "Rezidivnävus", ["NV"]),
    (r"^growing nevi$", "Wachsende Nävi", ["NV"]), (r"^congenital nevi$|^congenital melanocytic nevus$", "Kongenitale Nävi", ["NV"]),
    (r"hypermelanotic clark \(dysplastic\) nevus", "Dysplastischer Nävus", ["NV"]),
    (r"^melanocytic lesions?$|^flat melanocytic lesions$", "Melanozytäre Läsionen (unspezifisch)", ["NV", "MEL"]),
    (r"^facial pigmented lesions$", "Pigmentierte Gesichtsläsionen (unspezifisch)", ["NV", "MEL", "BKL"]),
    (r"^dermatofibroma$", "Dermatofibrom", ["DF"]), (r"^hemangioma$", "Hämangiom", ["VASC"]),
    (r"^bowen disease$|^pigmented bowen disease$", "Bowen-Krankheit", ["AKIEC"]), (r"^scc$", "Plattenepithelkarzinom", ["AKIEC"]),
    (r"^actinic keratosis$", "Aktinische Keratose", ["AKIEC"]), (r"actinic-damaged skin", "Aktinisch geschädigte Haut", ["AKIEC"]),
    (r"^viral warts$", "Viruswarzen", ["OTHER"]), (r"^sebaceous hyperplasia$", "Talgdrüsenhyperplasie", ["OTHER"]),
    (r"^clear cell acanthoma$", "Klarzellakanthom", ["OTHER"]), (r"^inflammatory diseases$", "Entzündliche Dermatosen", ["OTHER"]),
]
SPECIAL = {  # whole-string overrides (checked first)
    "May indicate malignancy in appropriate context, e.g. flat melanocytic lesions": [("Hinweis auf Malignität (flache melanozytäre Läsion)", ["MEL"])],
    "Various diagnoses, but mainly actinic keratosis; SCC; actinic-damaged skin": [("Aktinische Keratose", ["AKIEC"]), ("Plattenepithelkarzinom", ["AKIEC"]), ("Aktinisch geschädigte Haut", ["AKIEC"])],
}
GENERIC = re.compile(r"^(various diagnoses|\(7-point checklist|regular: )", re.I)


def parse(text):
    """-> (list of (display, groups), generic?)"""
    text = (text or "").strip()
    if not text or GENERIC.match(text):
        return [], bool(text)
    if text in SPECIAL:
        return SPECIAL[text], False
    out = []
    for tok in re.split(r";\s*", text):
        t = re.sub(r"^(centric|eccentric): ", "", tok.strip().lower())
        for pat, name, groups in TOK:
            if re.search(pat, t):
                out.append((name, groups)); break
        else:
            raise ValueError(f"unmapped diagnosis token: {tok!r} in {text!r}")
    return out, False


def load(path="taxonomy.csv"):
    """-> {ID: dict(entries=[(name, groups)], groups=[...], generic=bool)}"""
    res = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        e, gen = parse(r["Associated diagnoses (as listed in source)"])
        g = sorted({x for _, gs in e for x in gs})
        res[r["ID"]] = dict(entries=e, groups=g, generic=gen)
    return res


def weights(m):
    """{ID: {group: weight}} with weight = 1/#groups the ID lists (0 for generic / unlisted)."""
    return {i: {g: 1 / len(v["groups"]) for g in v["groups"]} for i, v in m.items() if v["groups"]}


def scores(conf, m=None, normalise=False):
    """conf: {ID: confidence 0-1 (already thresholded)} -> {group: evidence score}; normalise divides by the number of IDs listing the group."""
    m = m or load(); w = weights(m)
    n_ids = {g: sum(1 for i in w if g in w[i]) for g in GROUPS}
    s = {g: 0.0 for g in GROUPS}
    for i, c in conf.items():
        for g, x in w.get(i, {}).items():
            s[g] += c * x
    return {g: (v / n_ids[g] if normalise and n_ids[g] else v) for g, v in s.items()}


if __name__ == "__main__":
    m = load(); w = weights(m)
    import collections
    print("IDs with diagnoses:", len(w), "| generic only:", sum(1 for v in m.values() if v["generic"]), "| none:", sum(1 for v in m.values() if not v["groups"] and not v["generic"]))
    n = collections.Counter(g for i in w for g in w[i]); print("IDs listing each group:", {g: n[g] for g in GROUPS})
    for i in sorted(m):
        if m[i]["groups"]:
            print(i, [n_ for n_, _ in m[i]["entries"]], m[i]["groups"])
