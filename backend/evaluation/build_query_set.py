"""Build the W5 golden query set from real usage in backend/data/app.db.

The query set is *sourced*, not invented: every entry is the verbatim text of a
`messages` row with `role='user'`. This module holds the curation decisions
(which message IDs were kept, which were dropped and why, what kind each query
is, and which high-signal literal terms it carries) so the set is reproducible
and auditable rather than a hand-typed JSON blob.

Run:
    python -m evaluation.build_query_set            # writes golden_queries_v1.json
    python -m evaluation.build_query_set --check    # verify the JSON matches the DB
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

VERSION = "v1"
EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DB = EVAL_DIR.parent / "data" / "app.db"
DEFAULT_OUT = EVAL_DIR / f"golden_queries_{VERSION}.json"


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------
# `KEPT` maps the source `messages.id` (the lowest id among identical texts) to
# the curation metadata. `kind` is the stratum used for the per-kind breakdown:
#
#   factual          - a specific fact, value or mechanism is being asked for
#   methods          - a protocol, procedure, reagent or condition
#   library_filtered - scoped to a subset of the library (author, title, DOI,
#                      "do any of my X papers...")
#   synthesis        - framing, ideation, cross-paper summarisation
#
# `terms` is a list of variant-groups. A group is "present" in a chunk when any
# of its variants appears (case-insensitive, after unicode sub/superscript
# folding). Only terms that literally occur in the query are listed - this is a
# *literal-presence* sanity metric, not a semantic one. An empty list means the
# query carries no distinctive term and is skipped by the term metrics.

KEPT: Dict[int, Dict[str, Any]] = {
    1: {
        "kind": "synthesis",
        "terms": [["LXXLL", "LxxLL"], ["SRC-2", "SRC2", "SRC 2", "NCOA2", "steroid receptor coactivator"]],
    },
    3: {
        "kind": "synthesis",
        "terms": [["SRC2", "SRC-2", "SRC 2"], ["SRC3", "SRC-3", "SRC 3", "NCOA3"], ["coactivator"]],
    },
    9: {
        "kind": "factual",
        "terms": [["SRC2", "SRC-2", "SRC 2", "SRC"]],
    },
    11: {
        "kind": "factual",
        "terms": [["SRC2", "SRC-2", "SRC 2", "SRC"], ["potency", "potent", "IC50", "Ki", "Kd"]],
    },
    112: {
        "kind": "library_filtered",
        "terms": [["A'Lester", "ALester", "Allen"], ["SERS", "surface-enhanced Raman", "surface enhanced Raman"]],
    },
    124: {
        "kind": "factual",
        "terms": [["apidaecin"], ["FITC", "fluorescein isothiocyanate", "fluorescein-5-isothiocyanate"]],
    },
    130: {
        "kind": "methods",
        "terms": [
            ["tetramethylguanidin", "tetramethyl guanidin", "guanidinylat", "TMG"],
            ["N-terminus", "N terminus", "N-terminal", "N terminal"],
        ],
    },
    133: {
        "kind": "methods",
        "terms": [["apidaecin"], ["Api137", "Api 137", "Api-137", "apidaecin 137"], ["global cleavage", "cleavage cocktail", "TFA cleavage"]],
    },
    136: {
        "kind": "methods",
        "terms": [["apidaecin"], ["ornithine", "Orn"], ["global deprotection", "side-chain deprotection", "deprotection"]],
    },
    145: {
        "kind": "methods",
        "terms": [["peptide cleavage", "cleavage time", "cleavage cocktail"]],
    },
    154: {
        "kind": "synthesis",
        "terms": [["peptide therapeutic", "therapeutic peptide"]],
    },
    169: {
        "kind": "synthesis",
        "terms": [
            ["estrogen receptor", "oestrogen receptor", "ERalpha", "ER-alpha", "ESR1", "ERα"],
            ["breast cancer"],
            ["peptide inhibitor"],
        ],
    },
    177: {
        "kind": "factual",
        "terms": [
            ["intracellular concentration", "cellular concentration", "intracellular accumulation"],
            ["limit of detection", "detection limit", "LOD"],
            ["SRS", "stimulated Raman"],
        ],
    },
    181: {
        "kind": "library_filtered",
        "terms": [["Wei Min", "Min W"], ["stimulated Raman scattering", "SRS"]],
    },
    185: {
        "kind": "library_filtered",
        "terms": [
            ["10.1038/s41566-025-01707-z"],
            ["stimulated Raman scattering", "SRS"],
            ["Wei Min", "Ji-Xin Cheng", "Ozeki"],
        ],
    },
    187: {
        "kind": "factual",
        "terms": [
            ["MIC", "minimum inhibitory concentration", "minimal inhibitory concentration"],
            ["apidaecin"],
            ["Api137", "Api 137", "Api-137"],
        ],
    },
    189: {
        "kind": "factual",
        "terms": [
            ["LSPR", "localized surface plasmon", "localised surface plasmon"],
            ["hollow nanoparticle", "hollow gold", "nanoshell"],
            ["SERS"],
        ],
    },
    193: {
        "kind": "synthesis",
        "terms": [["plasmonic"], ["SERS"], ["SARS-CoV-2", "SARS CoV 2", "COVID-19", "coronavirus"]],
    },
    195: {
        "kind": "factual",
        "terms": [["SRC2-SP4", "SRC2 SP4", "SP4"], ["R4K1"]],
    },
    197: {
        "kind": "synthesis",
        "terms": [["nuclear receptor"], ["LXXLL", "LxxLL"]],
    },
    199: {
        "kind": "library_filtered",
        "terms": [["medicinal chemistry", "med chem", "J. Med. Chem", "ACS Med"]],
    },
    201: {
        "kind": "synthesis",
        "terms": [["target class"], ["modality", "modalities"]],
    },
    203: {
        "kind": "synthesis",
        "terms": [["peptide therap"]],
    },
    205: {
        "kind": "library_filtered",
        "terms": [["Schepartz"]],
    },
    207: {
        "kind": "synthesis",
        "terms": [["beta peptide", "beta-peptide", "β-peptide", "β peptide"], ["LXXLL", "LxxLL"]],
    },
    209: {
        "kind": "methods",
        "terms": [["hydrocarbon staple", "stapled peptide", "stapling"], ["salt bridge"], ["side chain", "side-chain"]],
    },
    211: {
        "kind": "factual",
        "terms": [["stimulated Raman scattering", "SRS"], ["concentration"]],
    },
    213: {
        "kind": "library_filtered",
        "terms": [["SRS", "stimulated Raman"], ["Conor Evans", "Evans"], ["skin"], ["detection limit", "limit of detection"]],
    },
    215: {
        "kind": "methods",
        "terms": [["tetramethylguanidin", "tetramethyl guanidin", "guanidinylat", "TMG"]],
    },
    217: {
        "kind": "methods",
        "terms": [["Glaser coupling", "Glaser-Hay", "Glaser hay", "Glaser"], ["off-resin", "off resin", "in solution", "solution-phase"]],
    },
    219: {
        "kind": "methods",
        "terms": [
            ["apidaecin"],
            ["propargyl glycine", "propargylglycine", "Pra"],
            ["tetramethylguanidino", "guanidino", "guanidinylat"],
            ["MALDI", "MALDI-TOF"],
        ],
    },
    221: {
        "kind": "methods",
        "terms": [
            ["Glaser coupling", "Glaser-Hay", "Glaser"],
            ["CuCl", "copper(I) chloride", "cuprous chloride"],
            ["bipyridine", "bipy", "2,2'-bipyridine"],
            ["propargyl serine", "propargylserine"],
        ],
    },
    229: {
        "kind": "methods",
        "terms": [
            ["diisopropylcarbodiimide", "DIC", "DIPCDI"],
            ["Oxyma"],
            ["SPPS", "solid-phase peptide synthesis", "solid phase peptide synthesis"],
            ["microwave"],
        ],
    },
    231: {
        "kind": "library_filtered",
        "terms": [
            ["TMEDA", "tetramethylethylenediamine", "N,N,N',N'-tetramethylethylenediamine"],
            ["CuCl", "copper(I) chloride"],
            ["copper catalysis", "Cu catalysis", "copper-catalyzed", "copper-catalysed"],
        ],
    },
    233: {
        "kind": "methods",
        "terms": [
            ["Glaser coupling", "Glaser-Hay", "Glaser"],
            ["CuCl", "copper(I) chloride"],
            ["TMEDA", "tetramethylethylenediamine"],
            ["DCM", "dichloromethane"],
        ],
    },
    239: {
        "kind": "methods",
        "terms": [
            ["2-chlorotrityl", "chlorotrityl", "2-CTC", "CTC resin"],
            ["Sieber", "Sieber amide"],
            ["acid labile", "acid-labile"],
        ],
    },
    245: {
        "kind": "factual",
        "terms": [["antimicrobial peptide", "AMP"], ["Gram-positive", "gram positive", "Gram +"]],
    },
    247: {
        "kind": "factual",
        "terms": [["E. coli", "E.coli", "Escherichia coli"], ["skin"]],
    },
    249: {
        "kind": "methods",
        "terms": [["E. coli", "E.coli", "Escherichia coli"], ["skin infection", "skin"]],
    },
    251: {
        "kind": "methods",
        "terms": [
            ["estrogen receptor", "ERα", "ERalpha", "ESR1"],
            ["LBD", "ligand binding domain", "ligand-binding domain"],
            ["D538G"],
            ["Y537S"],
            ["plasmid", "expression system", "E. coli expression", "baculovirus"],
        ],
    },
    253: {
        "kind": "synthesis",
        "terms": [
            ["azide", "azido"],
            ["antimicrobial peptide", "AMP"],
            ["mRNA display"],
            ["BADY"],
            ["photothermal"],
        ],
    },
    257: {
        "kind": "methods",
        "terms": [
            ["propargyl glycine", "propargylglycine", "Fmoc-Pra"],
            ["phenylacetylene"],
            ["diynoic acid", "hepta-4,6-diynoic", "diyne"],
        ],
    },
    259: {
        "kind": "factual",
        "terms": [["diyne", "diynyl", "butadiyne"], ["IR active", "infrared", "IR-active"]],
    },
    261: {
        "kind": "factual",
        "terms": [["Raman cross-section", "Raman cross section", "cross-section"], ["alkyne", "nitrile", "deuterium"]],
    },
    263: {
        "kind": "methods",
        "terms": [
            ["PhDY", "phenyl-diyne", "phenylhepta-4,6-diynoic"],
            ["BADY", "phenylbuta-1,3-diyn"],
            ["Fmoc"],
        ],
    },
    265: {
        "kind": "methods",
        "terms": [
            ["triethylamine", "Et3N", "TEA", "NEt3"],
            ["Pd(PPh3)2Cl2", "PdCl2(PPh3)2", "bis(triphenylphosphine)palladium", "Pd(PPh₃)₂Cl₂"],
        ],
    },
    267: {
        "kind": "factual",
        "terms": [["apidaecin"], ["intracellular", "inside cells", "accumulat"], ["concentration"]],
    },
    269: {
        "kind": "library_filtered",
        "terms": [["intracellular concentration", "intracellular accumulation"], ["peptide"]],
    },
    271: {
        "kind": "library_filtered",
        "terms": [
            ["Glaser"],
            ["Glaser-Hay", "Glaser Hay"],
            ["Cadiot-Chodkiewicz", "Cadiot Chodkiewicz", "Chodkiewicz"],
            ["Tetrakis(triphenylphosphine)palladium", "Pd(PPh3)4", "Pd(PPh₃)₄", "tetrakis"],
        ],
    },
    273: {
        "kind": "synthesis",
        "terms": [],
    },
}


# Dropped message IDs and why. Kept in the artifact so the selection is
# falsifiable - anyone can re-read these and disagree.
DROPPED: Dict[int, str] = {
    5: "library-metadata question (paper counts, top authors); answered by aggregation over the papers table, not by chunk retrieval - no chunk can be relevant",
    65: "byte-duplicate of id 1 with trailing whitespace",
    115: "conversational fragment, no retrievable intent ('here are some of my publications')",
    118: "context-bound fragment - 'these papers' refers to an attached selection",
    121: "near-duplicate of id 112, which is the better-formed version of the same request",
    127: "meta-question about the previous answer, not about the library",
    139: "subsumed by id 215, which asks the same stoichiometry question with the protocol framing",
    142: "context-bound fragment ('how about for tbtu')",
    148: "presentation writing task: 5.7k-char job posting pasted in, no library information need",
    151: "title brainstorming, no library information need",
    157: "slide-title wording, no library information need",
    160: "7.5k-char slide-deck editing task",
    163: "'test'",
    165: "4.6k-char TED-talk transcript pasted as a style model",
    167: "talk-agenda structuring, no library information need",
    171: "prose insertion request into an existing draft",
    173: "title brainstorming",
    175: "abstract editing, pasted draft",
    179: "meta-question about a previous answer's sourcing",
    183: "near-duplicate of id 181 (same Wei Min SRS review)",
    191: "formula-definition request whose body is mangled unicode math; the query text itself is corrupted, so it is not a fair retrieval probe",
    223: "rewrite of an already-generated protocol - a generation task over prior output",
    225: "edit instruction on a previously generated protocol",
    227: "context-bound fragment ('what does the water do here')",
    235: "rewrite of an already-generated protocol",
    237: "leading edit instruction makes it context-bound",
    241: "edit instruction on a previously generated protocol",
    243: "context-bound fragment ('summarize these papers')",
    255: "context-bound follow-up ('go one by one on each project')",
    275: "conversational fragment ('yes please elaborate ...')",
}


def build(db_path: Path) -> Dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    total_user = conn.execute(
        "SELECT COUNT(*) c FROM messages WHERE role='user'"
    ).fetchone()["c"]
    total_conv = conn.execute(
        "SELECT COUNT(DISTINCT conversation_id) c FROM messages WHERE role='user'"
    ).fetchone()["c"]

    # Occurrence count for each distinct text, keyed by the lowest message id.
    dupes = {}
    for row in conn.execute(
        "SELECT MIN(id) mid, COUNT(*) n, COUNT(DISTINCT conversation_id) nc "
        "FROM messages WHERE role='user' GROUP BY trim(lower(content))"
    ):
        dupes[row["mid"]] = (row["n"], row["nc"])

    queries: List[Dict[str, Any]] = []
    for i, (msg_id, meta) in enumerate(sorted(KEPT.items()), start=1):
        row = conn.execute(
            "SELECT content, conversation_id FROM messages WHERE id=?", (msg_id,)
        ).fetchone()
        if row is None:
            raise SystemExit(f"message id {msg_id} not found in {db_path}")
        n, nc = dupes.get(msg_id, (1, 1))
        queries.append(
            {
                "query_id": f"q{i:02d}",
                "text": row["content"],
                "kind": meta["kind"],
                "terms": meta["terms"],
                "source_message_id": msg_id,
                "source_conversation_id": row["conversation_id"],
                "occurrences": n,
                "conversations": nc,
            }
        )
    conn.close()

    by_kind: Dict[str, int] = {}
    for q in queries:
        by_kind[q["kind"]] = by_kind.get(q["kind"], 0) + 1

    return {
        "version": VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "db": str(db_path),
            "table": "messages",
            "filter": "role='user'",
            "total_user_messages": total_user,
            "total_conversations": total_conv,
            "distinct_texts": len(dupes),
        },
        "composition": {
            "n_queries": len(queries),
            "by_kind": by_kind,
            "n_dropped_distinct": len(DROPPED),
        },
        "dropped": {str(k): v for k, v in sorted(DROPPED.items())},
        "queries": queries,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="verify the existing JSON still matches the DB")
    args = ap.parse_args()

    data = build(args.db)

    if args.check:
        existing = json.loads(args.out.read_text())
        old = {q["query_id"]: q["text"] for q in existing["queries"]}
        new = {q["query_id"]: q["text"] for q in data["queries"]}
        if old != new:
            raise SystemExit("MISMATCH: golden query texts differ from the database")
        print(f"OK: {len(new)} queries match {args.db}")
        return

    args.out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Wrote {len(data['queries'])} queries to {args.out}")
    for k, v in sorted(data["composition"]["by_kind"].items()):
        print(f"  {k:18s} {v}")
    print(f"  dropped (distinct)  {len(DROPPED)}")


if __name__ == "__main__":
    main()
