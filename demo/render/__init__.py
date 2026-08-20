"""
Shared HTML renderers for the demo pages (pipeline_report.py, llm_report.py).

Every `stage_*` function takes a `case` dict from `demo.cases.load_case(..., trace=True)`
and returns an HTML fragment. The page is a WALKTHROUGH of the pipeline in the
order the stages actually run, so the panels here mirror `graph/pipeline.py`.

WHERE THINGS LIVE. This was one 1300-line module; it is now one package with the
same public surface:

    primitives.py       small formatting helpers, palette + label tables
    provenance.py       action badges, deterministic-check lists, review flags
    stages_cluster.py   detect -> name clustering -> coref -> relations
    stages_people.py    interviewee identification + per-person cards
    stages_world.py     places, dates/ages, identifiers
    stages_graph.py     entity-graph SVG, edge table, ledger, artifact
    page.py             metrics tiles, stepper, `transcript_panel`
    css.py              the stylesheet

Dependencies run strictly downward in that order; there are no cycles.
The two report scripts use only `transcript_panel` and `CSS`.
"""

from .primitives import (_HL, _PERSON_FILL, _ID_CATS, _DATE_CATS, _ID_LABEL,
                         _relname, _pname, _pct, _val, section, _chip, _names_map)
from .provenance import (_ACTION, _action_badge, _checks_html, _prov_of,
                         _prov_table, _prov_list, _prov_details, _flag_items,
                         _flag_html)
from .stages_cluster import stage_detect, stage_cluster, stage_coref, stage_relations
from .stages_people import stage_interviewee, stage_people
from .stages_world import stage_places, stage_dates_ages, stage_identifiers
from .stages_graph import stage_graph, stage_ledger, stage_artifact
from .page import metrics_grid, transcript_panel
from .css import CSS

__all__ = [
    "section", "metrics_grid", "transcript_panel", "CSS",
    "stage_detect", "stage_cluster", "stage_coref", "stage_relations",
    "stage_interviewee", "stage_people", "stage_places", "stage_dates_ages",
    "stage_identifiers", "stage_graph", "stage_ledger", "stage_artifact",
]
