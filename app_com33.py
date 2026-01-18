from __future__ import annotations

import csv
import re
import html
import tempfile
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import graphviz
from fpdf import FPDF

# ============================================================
# COMBINED APP CONFIG
# ============================================================
COMBINED_APP_NAME = "Prostate Navigator (Screening → Post-biopsy)"
COMBINED_APP_VERSION = "0.1.0"
COMBINED_APP_AUTHOR = "Mahziar Khazaali, MD"
COMBINED_COPYRIGHT_YEAR = "2026"

st.set_page_config(page_title=COMBINED_APP_NAME, page_icon="🩺", layout="wide")

# NOTE: This file merges:
# - AUA/SUO Screening Wizard (from app_scre85.py)
# - Post-biopsy Risk Navigator (from app6g.py)
# Transition: Screening Step 3 (Biopsy result) → if Positive → Post-biopsy module

# ============================================================
# APP CONFIG
# ============================================================
APP_NAME = "AUA/SUO Prostate Screening Navigator"
APP_VERSION = "1.9.0"
APP_AUTHOR = "Mahziar Khazaali, MD"
COPYRIGHT_YEAR = "2026"


# ============================================================
# DEFAULTS / STATE
# ============================================================
DEFAULTS = {
    # Wizard
    "wizard_step": 1,  # 1=Screening, 2=Evaluation, 3=Biopsy result, 4=After negative biopsy, 5=Risk Stratification

    # Patient
    "age": 55,
    "life_expectancy_10y": True,

    # Risk factors
    "black_ancestry": False,
    "germline_mutation": False,
    "strong_fh": False,

    # PSA/DRE
    "psa_done": True,
    "psa_value": 2.5,
    "psa_repeated": False,
    "psa_repeat_value": 0.0,
    # Confirmation flags: repeat PSA should only drive downstream decisions once it is
    # explicitly recorded (saved) or auto-saved when advancing steps.
    "psa_initial_confirmed": False,
    "psa_repeat_confirmed": False,
    "psa_initial_saved": None,
    "psa_repeat_saved": None,
    "psa_date": None,
    "psa_repeat_date": None,
    "psa_history": [],
    "dre_abnormal": False,
    
    # Calculator Inputs
    "psa_free": 0.0,
    "prostate_vol": 40.0,

    # Advanced diagnostics
    "mri_done": False,
    "pirads": "No MRI",
    "psa_density": 0.10,
    "psa_density_saved": None,
    "biomarker_done": False,
    "biomarker_positive": False,
    "biomarker_entries": [],

    # Biopsy
    "biopsy_result": "Not done",  # Not done / Negative / Positive
    "biopsy_date": None,
    
    # Risk Stratification Inputs
    "clinical_t": "cT1c",
    "grade_group": 1,
    "positive_cores": 20,

    # Calculated Outputs
    "calc_psa_density": 0.0,
    "calc_free_psa_pct": 0.0,
    "risk_group": None

    # Sticky snapshot (internal)
    ,"_sticky": None
}

PSA_DENSITY_HIGH = 0.15


def init_state() -> None:
    # Use deepcopy for mutable defaults (lists/dicts)
    import copy
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = copy.deepcopy(v)

    # Date defaults
    if st.session_state.get('psa_date') is None:
        st.session_state['psa_date'] = datetime.date.today()
    if st.session_state.get('psa_repeat_date') is None:
        st.session_state['psa_repeat_date'] = datetime.date.today()
    if st.session_state.get('biopsy_date') is None:
        st.session_state['biopsy_date'] = datetime.date.today()
    if not isinstance(st.session_state.get('psa_history'), list):
        st.session_state['psa_history'] = []


def _sticky_restore() -> None:
    """Restore key screening inputs if they unexpectedly snap back to defaults.

    In some Streamlit environments, values associated with widgets that are not
    rendered on a particular step can revert to defaults. We keep an internal
    snapshot (st.session_state['_sticky']) and restore from it when we detect
    a snap-back.

    We only restore when the current value equals the app DEFAULT and the
    snapshot has a different (non-None) value.
    """
    s = st.session_state
    snap = s.get("_sticky", None)
    if not isinstance(snap, dict):
        return

    for k, snap_v in snap.items():
        if k not in DEFAULTS:
            continue
        if snap_v is None:
            continue
        try:
            live_v = s.get(k, DEFAULTS[k])
            default_v = DEFAULTS[k]
            # Restore only when live == default and snapshot != default
            if live_v == default_v and snap_v != default_v:
                s[k] = snap_v
        except Exception:
            # If comparison fails (rare), avoid crashing the app.
            continue


def _sticky_update() -> None:
    """Update the internal snapshot for key inputs.

    - Step 1: capture demographic/risk + initial PSA/DRE.
    - Step 2: capture repeat PSA settings.

    IMPORTANT: We do NOT update snapshot on other steps to avoid overwriting it
    with defaults if Streamlit snaps values back.
    """
    s = st.session_state
    step = int(s.get("wizard_step", 1) or 1)
    snap = s.get("_sticky", None)
    if not isinstance(snap, dict):
        snap = {}

    if step == 1:
        for k in [
            "age",
            "life_expectancy_10y",
            "black_ancestry",
            "germline_mutation",
            "strong_fh",
            "psa_done",
            "psa_value",
            "dre_abnormal",
        ]:
            snap[k] = s.get(k)

    if step == 2:
        for k in ["psa_repeated", "psa_repeat_value"]:
            snap[k] = s.get(k)

    s["_sticky"] = snap


def reset_all() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    # Also clear internal snapshots so "Reset" truly resets.
    st.session_state["_sticky"] = None
    st.session_state["psa_initial_saved"] = None
    st.session_state["psa_repeat_saved"] = None
    set_qp({})
    st.rerun()


# ============================================================
# QUERY PARAMS (for focus/zoom window)
# ============================================================
def get_qp() -> dict:
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()


def set_qp(params: dict) -> None:
    try:
        st.query_params.clear()
        for k, v in params.items():
            st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**params)


def qp_first(qp: dict, key: str, default: str = "") -> str:
    v = qp.get(key, default)
    if isinstance(v, list):
        return v[0] if v else default
    return v if v is not None else default


# ============================================================
# CLINICAL HELPERS / CALCULATORS
# ============================================================
def calculate_psa_density(psa, vol):
    if vol <= 0: return 0.0
    return psa / vol

def calculate_free_psa_pct(free, total):
    if total <= 0: return 0.0
    return (free / total) * 100.0


def _cb_calc_psad_from_inputs() -> None:
    """Callback-safe PSAD calculator.

    Streamlit does not allow assigning to a widget-keyed session_state value
    *after* that widget is instantiated in the same run. Using an on_click
    callback updates state *before* the rerun renders widgets.
    """
    # Streamlit runs callbacks at the *start* of a rerun. In a fresh session,
    # default keys (e.g., psa_done) may not exist yet unless we initialize here.
    init_state()
    psa = current_psa() or 0.0
    try:
        vol = float(st.session_state.get("prostate_vol", 0.0) or 0.0)
    except Exception:
        vol = 0.0
    res = calculate_psa_density(psa, vol)
    st.session_state["calc_psa_density"] = res
    st.session_state["psa_density"] = res
    st.session_state["psa_density_saved"] = res
    st.session_state["_psad_just_calculated"] = True

def calculate_nccn_risk(t_stage, grade, psa, cores_pct):
    # Simplified NCCN Logic for Post-Biopsy Stratification
    if "T3b" in t_stage or "T4" in t_stage or grade == 5:
        return "Very High"
    if "T3a" in t_stage or grade == 4 or psa > 20:
        return "High"
    
    is_inter = False
    if "T2b" in t_stage or "T2c" in t_stage or grade in [2,3] or (10 <= psa <= 20):
        is_inter = True
    
    if is_inter:
        factors = 0
        if "T2b" in t_stage or "T2c" in t_stage: factors += 1
        if grade in [2,3]: factors += 1
        if 10 <= psa <= 20: factors += 1
        
        if grade == 3 or cores_pct >= 50 or factors >= 2:
            return "Unfavorable Intermediate"
        return "Favorable Intermediate"
        
    return "Low"

def is_high_risk() -> bool:
    s = st.session_state
    return bool(s.black_ancestry or s.germline_mutation or s.strong_fh)


def psa_threshold(age: int) -> float:
    # Simplified common age-based thresholds (for tool behavior)
    if age < 50:
        return 2.5
    if age < 60:
        return 3.5
    if age < 70:
        return 4.5
    return 6.5


def current_psa() -> float | None:
    """Return the PSA value that should drive the screening/evaluation flow.

    Rules:
    - If PSA not performed → None
    - If repeat PSA is marked as done → use repeat PSA (prefer saved repeat if a bug resets UI values)
    - Otherwise → use initial PSA (prefer saved initial if a bug resets UI values)

    This is intentionally defensive: users reported that PSA can revert to the
    default value on some reruns. We therefore preserve the last user-entered
    PSA(s) in dedicated session keys and fall back to them when needed.
    """
    s = st.session_state
    if not bool(s.get("psa_done", False)):
        return None

    def _to_float(x) -> float | None:
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    # ---- Initial PSA (baseline) ----
    val = _to_float(s.get("psa_value", None))
    saved = _to_float(s.get("psa_initial_saved", None))
    if val is None:
        baseline = saved
    else:
        # If UI value snaps back to the default but we have a different saved initial, trust saved.
        if saved is not None and val == float(DEFAULTS["psa_value"]) and saved != val:
            baseline = saved
        else:
            baseline = val

    # ---- Repeat PSA (becomes active once a real value is entered) ----
    # IMPORTANT: Checking "PSA repeated" alone should NOT overwrite the
    # current PSA. We only switch to repeat PSA once the value is > 0.
    if bool(s.get("psa_repeated", False)):
        rep_saved = _to_float(s.get("psa_repeat_saved", None))
        if rep_saved is not None and rep_saved > 0:
            return rep_saved
        rep_val = _to_float(s.get("psa_repeat_value", None))
        if rep_val is not None and rep_val > 0:
            return rep_val

    return baseline



def _psa_persistence_tick() -> None:
    """Persist user-entered PSA values across wizard navigation.

    Problem observed: moving between steps (especially Step 2) can cause PSA
    (and sometimes risk context) shown in the sidebar to revert to defaults.

    Strategy:
    - Save the last user-entered initial PSA into psa_initial_saved.
    - Save the last user-entered repeat PSA into psa_repeat_saved when repeat is enabled.
    - On steps where the PSA widgets are NOT shown, if the live PSA value snaps
      back to the DEFAULT but a saved value exists, restore the live value.

    This keeps the rest of the app (which reads psa_value/psa_repeat_value)
    consistent without changing the UI layout.
    """
    s = st.session_state
    step = int(s.get("wizard_step", 1) or 1)

    if bool(s.get("psa_done", False)):
        # --- Initial PSA ---
        try:
            cur = s.get("psa_value", None)
            if cur is not None:
                cur_f = float(cur)
                # Always capture at least once; update when Step 1 is visible (where PSA input is shown)
                if s.get("psa_initial_saved", None) is None or step == 1:
                    s["psa_initial_saved"] = cur_f
        except Exception:
            pass

        # If we're NOT on Step 1 (PSA input is not rendered) and PSA snaps back to default, restore.
        if step != 1 and s.get("psa_initial_saved", None) is not None:
            try:
                live = float(s.get("psa_value", DEFAULTS["psa_value"]))
                saved = float(s.get("psa_initial_saved"))
                if live == float(DEFAULTS["psa_value"]) and saved != live:
                    s["psa_value"] = saved
            except Exception:
                pass

    # --- Repeat PSA (if enabled) ---
    if bool(s.get("psa_done", False)) and bool(s.get("psa_repeated", False)):
        try:
            rep = s.get("psa_repeat_value", None)
            if rep is not None:
                rep_f = float(rep)
                # Update when Step 2 is visible (repeat PSA input shown there)
                if s.get("psa_repeat_saved", None) is None or step == 2:
                    s["psa_repeat_saved"] = rep_f
        except Exception:
            pass

        # If we're NOT on Step 2 and repeat PSA snaps back to default, restore.
        if step != 2 and s.get("psa_repeat_saved", None) is not None:
            try:
                live = float(s.get("psa_repeat_value", DEFAULTS["psa_repeat_value"]))
                saved = float(s.get("psa_repeat_saved"))
                if live == float(DEFAULTS["psa_repeat_value"]) and saved != live:
                    s["psa_repeat_value"] = saved
            except Exception:
                pass

# ============================================================
# PSA HISTORY (date-stamped sidebar timeline)
# ============================================================

def _coerce_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _date_to_str(d) -> str:
    if d is None:
        return ''
    try:
        return d.isoformat()
    except Exception:
        return str(d)


def _append_psa_history(label: str, value, date_value) -> None:
    """Append a PSA entry to the PSA history (date + value + label).

    Avoids duplicate consecutive entries. Keeps a timeline (does not overwrite).
    """
    init_state()
    v = _coerce_float(value)
    if v is None:
        return
    # Repeat PSA widget defaults can be 0.00 (meaning "not entered").
    # Do not record/confirm repeat PSA unless it's a real value.
    if label == 'Repeat PSA' and v <= 0:
        return
    d = _date_to_str(date_value)
    entry = {'date': d, 'psa': v, 'label': label}

    hist = st.session_state.get('psa_history')
    if not isinstance(hist, list):
        hist = []

    if hist and hist[-1] == entry:
        st.session_state['psa_history'] = hist
        return

    hist.append(entry)
    st.session_state['psa_history'] = hist

    # Also persist the last recorded PSA values in dedicated keys.
    # These are used to prevent "snap back" to defaults and to control when
    # repeat PSA actually becomes the active PSA for decision-making.
    if label == 'PSA':
        st.session_state['psa_initial_saved'] = v
        st.session_state['psa_initial_confirmed'] = True
    elif label == 'Repeat PSA':
        st.session_state['psa_repeat_saved'] = v
        st.session_state['psa_repeat_confirmed'] = True


def _cb_save_initial_psa() -> None:
    init_state()
    if not bool(st.session_state.get('psa_done', False)):
        return
    _append_psa_history('PSA', st.session_state.get('psa_value'), st.session_state.get('psa_date'))


def _cb_save_repeat_psa() -> None:
    init_state()
    if not (bool(st.session_state.get('psa_done', False)) and bool(st.session_state.get('psa_repeated', False))):
        return
    _append_psa_history('Repeat PSA', st.session_state.get('psa_repeat_value'), st.session_state.get('psa_repeat_date'))



def _iso_date(d) -> str:
    try:
        return d.isoformat()
    except Exception:
        return str(d)


def current_psad() -> float | None:
    """Best available PSA density (saved > calculated > current)."""
    try:
        v = st.session_state.get("psa_density_saved", None)
        if v is not None and float(v) > 0:
            return float(v)
    except Exception:
        pass
    try:
        v = float(st.session_state.get("calc_psa_density", 0.0) or 0.0)
        if v > 0:
            return v
    except Exception:
        pass
    try:
        v = float(st.session_state.get("psa_density", 0.0) or 0.0)
        if v > 0:
            return v
    except Exception:
        pass
    return None


def current_free_psa_pct() -> float | None:
    """Best available %free PSA (saved > calculated)."""
    try:
        v = st.session_state.get("psa_free_pct_saved", None)
        if v is not None and float(v) > 0:
            return float(v)
    except Exception:
        pass
    try:
        v = float(st.session_state.get("calc_free_psa_pct", 0.0) or 0.0)
        if v > 0:
            return v
    except Exception:
        pass
    return None


def _append_free_psa_history(free_value, date_val) -> None:
    init_state()
    try:
        free_v = float(free_value)
    except Exception:
        return
    if free_v <= 0:
        return

    total = current_psa()
    total_v = float(total) if total is not None else 0.0
    pct = calculate_free_psa_pct(free_v, total_v)

    entry = {
        "date": _iso_date(date_val),
        "total_psa": float(total_v) if total_v else None,
        "free_psa": float(free_v),
        "pct": float(pct),
    }

    hist = st.session_state.get("psa_free_history", [])
    if not isinstance(hist, list):
        hist = []

    # De-dup by same date + free_psa
    hist = [e for e in hist if not (e.get("date") == entry["date"] and float(e.get("free_psa", -1)) == entry["free_psa"]) ]
    hist.append(entry)
    try:
        hist = sorted(hist, key=lambda x: x.get("date", ""))
    except Exception:
        pass

    st.session_state["psa_free_history"] = hist
    st.session_state["psa_free_saved"] = float(free_v)
    st.session_state["psa_free_pct_saved"] = float(pct)
    st.session_state["calc_free_psa_pct"] = float(pct)


def _cb_save_free_psa() -> None:
    init_state()
    _append_free_psa_history(st.session_state.get("psa_free", 0.0), st.session_state.get("psa_free_date"))


def biomarker_high_risk() -> bool:
    entries = st.session_state.get("biomarker_entries", [])
    if not isinstance(entries, list):
        return False
    return any(bool(e.get("high_risk")) for e in entries)


def _add_biomarker_entry(name: str, date_val, result: str, interpretation: str) -> None:
    init_state()
    entries = st.session_state.get("biomarker_entries", [])
    if not isinstance(entries, list):
        entries = []

    high_risk = interpretation.strip().lower().startswith("high")
    entry = {
        "name": name,
        "date": _iso_date(date_val),
        "result": result,
        "interpretation": interpretation,
        "high_risk": bool(high_risk),
    }

    entries.append(entry)
    try:
        entries = sorted(entries, key=lambda x: x.get("date", ""))
    except Exception:
        pass

    st.session_state["biomarker_entries"] = entries

    # Keep legacy booleans in sync for existing logic
    st.session_state["biomarker_done"] = True
    st.session_state["biomarker_positive"] = biomarker_high_risk()


def _remove_biomarker_entry(idx: int) -> None:
    entries = st.session_state.get("biomarker_entries", [])
    if not isinstance(entries, list):
        return
    if 0 <= idx < len(entries):
        entries.pop(idx)
    st.session_state["biomarker_entries"] = entries
    st.session_state["biomarker_done"] = len(entries) > 0
    st.session_state["biomarker_positive"] = biomarker_high_risk()


def biomarker_panel_ui(context_key: str = "bm") -> None:
    """UI to add one or more biomarker results."""
    init_state()
    with st.expander("Adjunct biomarkers (optional)"):
        st.caption("Enter key result(s). Saved results will appear in the sidebar and report.")

        b1, b2 = st.columns([1.2, 1.0])
        with b1:
            bm_name = st.selectbox(
                "Biomarker",
                [
                    "PHI (Prostate Health Index)",
                    "4Kscore",
                    "ExoDx (EPI)",
                    "SelectMDx",
                    "PCA3",
                    "ConfirmMDx",
                    "Other",
                ],
                key=f"{context_key}_name",
            )
        with b2:
            bm_date = st.date_input("Date", key=f"{context_key}_date")

        # Minimal, high-yield fields per biomarker
        result_label = "Result"
        placeholder = ""
        if bm_name.startswith("PHI"):
            result_label = "PHI score"
            placeholder = "e.g., 42"
        elif bm_name.startswith("4K"):
            result_label = "4Kscore (% risk GG2+)"
            placeholder = "e.g., 12%"
        elif bm_name.startswith("ExoDx"):
            result_label = "ExoDx score"
            placeholder = "e.g., 18.6"
        elif bm_name.startswith("SelectMDx"):
            result_label = "SelectMDx result"
            placeholder = "Positive / Negative"
        elif bm_name.startswith("PCA3"):
            result_label = "PCA3 score"
            placeholder = "e.g., 35"
        elif bm_name.startswith("ConfirmMDx"):
            result_label = "ConfirmMDx result"
            placeholder = "Positive / Negative"
        elif bm_name == "Other":
            result_label = "Biomarker name + result"
            placeholder = "e.g., Stockholm3: 0.28 (elevated)"

        bm_result = st.text_input(result_label, placeholder=placeholder, key=f"{context_key}_result")
        bm_interp = st.selectbox(
            "Interpretation",
            ["Low-risk / reassuring", "Indeterminate", "High-risk / concerning"],
            key=f"{context_key}_interp",
        )

        cadd, cclear = st.columns(2)
        with cadd:
            if st.button("Add biomarker", key=f"{context_key}_add", use_container_width=True):
                if bm_result.strip() == "":
                    st.warning("Enter a result before adding.")
                else:
                    _add_biomarker_entry(bm_name, bm_date, bm_result.strip(), bm_interp)
                    st.success("Saved.")
        with cclear:
            if st.button("Clear all", key=f"{context_key}_clear", use_container_width=True):
                st.session_state["biomarker_entries"] = []
                st.session_state["biomarker_done"] = False
                st.session_state["biomarker_positive"] = False

        entries = st.session_state.get("biomarker_entries", [])
        if isinstance(entries, list) and entries:
            st.markdown("**Saved biomarker results**")
            for i, e in enumerate(entries):
                flag = "⚠️" if e.get("high_risk") else "✅"
                st.write(f"{flag} {e.get('date','')} — {e.get('name','')}: {e.get('result','')} ({e.get('interpretation','')})")
                st.button("Remove", key=f"{context_key}_rm_{i}", on_click=_remove_biomarker_entry, args=(i,))
        else:
            st.caption("No biomarker results saved.")

def _cb_save_active_psa() -> None:
    """Save the currently active PSA (repeat PSA if enabled)."""
    init_state()
    if not bool(st.session_state.get('psa_done', False)):
        return
    if bool(st.session_state.get('psa_repeated', False)):
        _cb_save_repeat_psa()
    else:
        _cb_save_initial_psa()



def psa_ge_2() -> bool:
    psa = current_psa()
    return bool(psa is not None and psa >= 2.0)


def psa_is_elevated_by_age_threshold() -> bool:
    psa = current_psa()
    if psa is None:
        return False
    return psa >= psa_threshold(int(st.session_state.age))


def mri_suspicious() -> bool:
    s = st.session_state
    return bool(s.mri_done and s.pirads in ["PI-RADS 3", "PI-RADS 4", "PI-RADS 5"])


def risk_params_high() -> bool:
    s = st.session_state
    density = s.calc_psa_density if s.calc_psa_density > 0 else float(s.psa_density)
    dre_risk = bool(s.dre_abnormal and psa_ge_2())
    return bool(
        is_high_risk()
        or biomarker_high_risk()
        or density > PSA_DENSITY_HIGH
        or dre_risk
    )


def screening_recommendation() -> tuple[str, str]:
    s = st.session_state
    age = int(s.age)

    if not s.life_expectancy_10y:
        return ("Discontinue routine screening.", "Screening benefit is lower when life expectancy is <10 years; individualize via SDM.")

    if age < 40:
        return ("Routine screening not generally recommended.", "Consider SDM only in unusual high-risk contexts.")
    if 40 <= age <= 45:
        if is_high_risk():
            return ("Offer PSA screening (SDM).", "Higher-risk groups may begin earlier.")
        return ("Routine screening not generally recommended.", "Consider baseline PSA in selected cases (SDM).")
    if 46 <= age <= 50:
        return ("Offer baseline PSA test (SDM).", "Baseline PSA can help future risk stratification.")
    if 51 <= age <= 69:
        return ("Offer regular screening (often every 2–4 years) using SDM.", "This age range has the strongest net benefit.")
    return ("Individualize or discontinue screening (SDM).", "Consider health status, PSA history, and preferences.")


def evaluation_recommendation() -> tuple[str, str]:
    s = st.session_state

    if s.dre_abnormal and not s.psa_done:
        return (
            "Obtain PSA first (DRE is supplementary; not a stand-alone screening test).",
            "Use PSA as the primary screening test, then interpret DRE as a risk-refining feature once PSA is known.",
        )

    if not s.psa_done:
        return ("Enter PSA to proceed.", "PSA is needed to place the patient on the evaluation pathway.")

    psa_elev = psa_is_elevated_by_age_threshold()

    if not psa_elev:
        if s.dre_abnormal:
            if not psa_ge_2():
                return (
                    "Abnormal DRE with PSA <2 → repeat PSA / safety-net follow-up (SDM).",
                    "DRE alone is not a screening test; when PSA is <2, the incremental risk signal is smaller. Confirm PSA and follow clinically.",
                )
            return (
                "PSA not elevated by age-threshold, but PSA ≥2 with abnormal DRE → risk evaluation (consider MRI ± biomarkers selectively).",
                "Abnormal DRE can refine risk during evaluation; use SDM to decide whether MRI/biomarkers would change biopsy decisions.",
            )
        return ("Resume screening / routine follow-up.", "PSA not elevated and DRE not suspicious.")

    if psa_elev and not s.psa_repeated:
        return ("Repeat PSA to confirm elevation.", "A newly elevated PSA should be confirmed before downstream testing.")

    if not s.mri_done:
        return ("Consider prostate MRI (± biomarkers selectively).", "MRI can better target biopsy and reduce unnecessary biopsy.")

    if mri_suspicious():
        return ("Proceed to targeted biopsy (± systematic).", "Suspicious MRI supports targeted sampling.")

    if risk_params_high():
        return ("Consider systematic biopsy (SDM).", "MRI negative, but risk parameters (including DRE when PSA ≥2) suggest residual risk.")
    return ("Forgo biopsy / safety-net follow-up (SDM).", "MRI negative and risk parameters low.")


def neg_biopsy_recommendation() -> tuple[str, str]:
    s = st.session_state
    if s.dre_abnormal and not s.psa_done:
        return (
            "Obtain PSA first (DRE is supplementary).",
            "Confirm PSA first; then use DRE as a risk-refining feature during evaluation.",
        )

    if not s.mri_done:
        return ("Obtain prostate MRI before considering repeat biopsy.", "MRI helps identify targets and stratify risk after a negative biopsy.")

    if mri_suspicious():
        return ("Repeat biopsy with MRI-targeted cores (± systematic).", "Suspicious MRI supports targeted re-biopsy.")

    if risk_params_high():
        return ("Consider repeat systematic biopsy (SDM).", "MRI negative but other risk parameters suggest residual risk.")
    return ("Resume screening / surveillance (SDM).", "MRI negative and low-risk parameters → lower probability of clinically significant cancer.")


# ============================================================
# PATHWAY DOT
# ============================================================
def node_style(nid: str, active: str) -> str:
    if nid == active:
        return 'style="rounded,filled" fillcolor="#dbeafe" color="#2563eb" fontcolor="#1e3a8a" penwidth="2.6"'
    return 'style="rounded,filled" fillcolor="white" color="#9ca3af" fontcolor="#374151" penwidth="1.2"'


def active_node_main() -> str:
    s = st.session_state
    if s.dre_abnormal and not s.psa_done: return "PSA_TEST"
    if not s.psa_done: return "PSA_TEST"
    psa_elev = psa_is_elevated_by_age_threshold()

    if not psa_elev:
        if s.dre_abnormal and psa_ge_2():
            return "GET_MRI" if not s.mri_done else ("TARGET_BIOPSY" if mri_suspicious() else ("SYS_BIOPSY" if risk_params_high() else "SDM_OMIT"))
        return "SCREEN_INTERVAL"

    if psa_elev and not s.psa_repeated: return "REPEAT_PSA"
    if not s.mri_done: return "GET_MRI"
    if mri_suspicious(): return "TARGET_BIOPSY"
    return "SYS_BIOPSY" if risk_params_high() else "SDM_OMIT"


def active_node_nb() -> str:
    s = st.session_state
    if not s.mri_done: return "NB_GET_MRI"
    if mri_suspicious(): return "NB_TARGET_BIOPSY"
    return "NB_SYS_BIOPSY" if risk_params_high() else "NB_RESUME"


def dot_main(active: str) -> str:
    return f"""
digraph MAIN {{
  rankdir=TB;
  bgcolor="transparent";
  graph [fontname="Helvetica" fontsize=22 pad="0.18" nodesep=0.55 ranksep=0.60 splines=ortho concentrate=true];
  node  [shape=box fontname="Helvetica" fontsize=16 margin="0.25,0.18" style="rounded,filled"];
  edge  [fontname="Helvetica" fontsize=12 color="#6b7280" penwidth=1.4 arrowsize=0.95];

  START [label="Start / risk\\nassessment" shape=ellipse {node_style("START", active)}];
  PSA_TEST [label="Obtain PSA\\n(primary test)" {node_style("PSA_TEST", active)}];
  DRE_NOTE [label="If abnormal DRE:\\nobtain/confirm PSA first" {node_style("DRE_NOTE", active)}];
  ELEVATED_Q [label="PSA elevated\\nby age-threshold?" shape=diamond style=filled fillcolor="#f3f4f6" color="#9ca3af"];
  SCREEN_INTERVAL [label="Resume screening\\n(2–4y via SDM)" {node_style("SCREEN_INTERVAL", active)}];
  REPEAT_PSA [label="Repeat PSA\\n(confirm elevation)" {node_style("REPEAT_PSA", active)}];
  GET_MRI [label="Consider / obtain MRI\\n(± biomarkers selective)" {node_style("GET_MRI", active)}];
  MRI_Q [label="MRI suspicious?\\n(PI-RADS ≥3)" shape=diamond style=filled fillcolor="#f3f4f6" color="#9ca3af"];
  TARGET_BIOPSY [label="Targeted biopsy\\n(± systematic)" {node_style("TARGET_BIOPSY", active)}];
  RISK_PARAMS [label="MRI 1–2 → risk params\\n(PSAD, biomarkers, DRE*)" {node_style("RISK_PARAMS", active)}];
  SYS_BIOPSY [label="Systematic biopsy\\n(selected cases)" {node_style("SYS_BIOPSY", active)}];
  SDM_OMIT [label="Forgo biopsy /\\nsafety net (SDM)" {node_style("SDM_OMIT", active)}];

  START -> PSA_TEST;
  START -> DRE_NOTE [style=dashed];
  PSA_TEST -> ELEVATED_Q;
  ELEVATED_Q -> SCREEN_INTERVAL [label="No"];
  ELEVATED_Q -> REPEAT_PSA [label="Yes (new)"];
  REPEAT_PSA -> ELEVATED_Q [label="Re-check"];
  ELEVATED_Q -> GET_MRI [label="Confirmed\\nelevation"];
  GET_MRI -> MRI_Q;
  MRI_Q -> TARGET_BIOPSY [label="Yes"];
  MRI_Q -> RISK_PARAMS [label="No (1–2)"];
  RISK_PARAMS -> SYS_BIOPSY [label="Higher risk"];
  RISK_PARAMS -> SDM_OMIT [label="Lower risk"];
}}
""".strip()


def dot_nb(active: str) -> str:
    return f"""
digraph NB {{
  rankdir=TB;
  bgcolor="transparent";
  graph [fontname="Helvetica" fontsize=22 pad="0.18" nodesep=0.55 ranksep=0.60 splines=ortho concentrate=true];
  node  [shape=box fontname="Helvetica" fontsize=16 margin="0.25,0.18" style="rounded,filled"];
  edge  [fontname="Helvetica" fontsize=12 color="#6b7280" penwidth=1.4 arrowsize=0.95];

  NB_START [label="After negative biopsy\\n(continue screening; re-assess)" shape=ellipse {node_style("NB_START", active)}];
  NB_GET_MRI [label="Obtain MRI\\n(if none) before repeat biopsy" {node_style("NB_GET_MRI", active)}];
  NB_MRI_Q [label="MRI suspicious?\\n(PI-RADS ≥3)" shape=diamond style=filled fillcolor="#f3f4f6" color="#9ca3af"];
  NB_TARGET_BIOPSY [label="Repeat targeted biopsy\\n(± systematic)" {node_style("NB_TARGET_BIOPSY", active)}];
  NB_RISK [label="MRI 1–2 → re-assess\\nrisk params (incl. DRE*)" {node_style("NB_RISK", active)}];
  NB_SYS [label="Consider repeat systematic\\nbiopsy (SDM)" {node_style("NB_SYS", active)}];
  NB_RESUME [label="Resume screening /\\nsurveillance (SDM)" {node_style("NB_RESUME", active)}];

  NB_START -> NB_GET_MRI;
  NB_GET_MRI -> NB_MRI_Q;
  NB_MRI_Q -> NB_TARGET_BIOPSY [label="Yes"];
  NB_MRI_Q -> NB_RISK [label="No (1–2)"];
  NB_RISK -> NB_SYS [label="Higher risk"];
  NB_RISK -> NB_RESUME [label="Lower risk"];
}}
""".strip()


# ============================================================
# ZOOM / FOCUS VIEW
# ============================================================
def render_dot_zoompan(dot: str, height: int, key: str) -> None:
    dot_js = dot.replace("\\", "\\\\").replace("`", "\\`")
    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  body {{ margin:0; padding:0; font-family: Helvetica, Arial, sans-serif; }}
  .toolbar {{ display:flex; gap:8px; align-items:center; padding:10px 12px; border-bottom:1px solid #e5e7eb; background:#fafafa; position: sticky; top: 0; z-index: 5; }}
  .toolbar button {{ padding:7px 11px; border:1px solid #d1d5db; border-radius:10px; background:white; cursor:pointer; font-size:14px; }}
  .viewer {{ height:{height}px; width:100%; overflow:hidden; background:white; }}
  .stage {{ width:100%; height:100%; }}
</style>
</head>
<body>
  <div class="toolbar">
    <button id="zin-{key}">+</button><button id="zout-{key}">-</button><button id="reset-{key}">Reset</button><button id="fit-{key}">Fit</button>
  </div>
  <div class="viewer" id="viewer-{key}"><div class="stage" id="stage-{key}"></div></div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/viz.js/2.1.2/viz.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/viz.js/2.1.2/full.render.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@panzoom/panzoom@4.5.1/dist/panzoom.min.js"></script>
  <script>
    const DOT = `{dot_js}`;
    const stage = document.getElementById("stage-{key}");
    const viewer = document.getElementById("viewer-{key}");
    function fitToView(pz, svgEl) {{
      const vb = svgEl.viewBox.baseVal;
      const vw = viewer.clientWidth;
      const vh = viewer.clientHeight;
      const scale = Math.min(vw / vb.width, vh / vb.height) * 0.97;
      pz.zoom(scale, {{ animate:false }});
      pz.pan((vw - vb.width * scale) / 2, (vh - vb.height * scale) / 2, {{ animate:false }});
    }}
    async function render() {{
      const viz = new Viz();
      try {{
        const svgEl = await viz.renderSVGElement(DOT);
        svgEl.style.display = "block";
        if (!svgEl.getAttribute("viewBox")) {{
           const w = parseFloat(svgEl.getAttribute("width") || "1200");
           const h = parseFloat(svgEl.getAttribute("height") || "900");
           svgEl.setAttribute("viewBox", `0 0 ${{w}} ${{h}}`);
        }}
        stage.appendChild(svgEl);
        const panzoom = Panzoom(svgEl, {{ maxScale: 8, minScale: 0.2, contain: "outside", cursor: "grab" }});
        viewer.addEventListener("wheel", panzoom.zoomWithWheel);
        document.getElementById("zin-{key}").onclick = () => panzoom.zoomIn();
        document.getElementById("zout-{key}").onclick = () => panzoom.zoomOut();
        document.getElementById("reset-{key}").onclick = () => {{ panzoom.reset({{ animate:false }}); fitToView(panzoom, svgEl); }};
        document.getElementById("fit-{key}").onclick = () => fitToView(panzoom, svgEl);
        setTimeout(() => fitToView(panzoom, svgEl), 40);
      }} catch (err) {{ stage.innerHTML = "Error: " + err; }}
    }}
    render();
  </script>
</body>
</html>
"""
    components.html(html, height=height + 56, scrolling=False)


def focus_mode() -> None:
    qp = get_qp()
    path = qp_first(qp, "path", "main")
    st.markdown(f"## 🗺️ Pathway (Zoom/Pan) — {('Main' if path=='main' else 'After negative biopsy')}")
    if st.button("⬅️ Back to wizard"):
        set_qp({})
        st.rerun()
    if path == "nb":
        render_dot_zoompan(dot_nb(active_node_nb()), 760, "focus-nb")
    else:
        render_dot_zoompan(dot_main(active_node_main()), 760, "focus-main")


# ============================================================
# WIZARD NAV
# ============================================================
def set_step(n: int) -> None:
    st.session_state.wizard_step = n
    st.rerun()

def sidebar_tracker():
    with st.sidebar:
        st.header("Patient Summary")
        st.caption("Mode: Screening wizard")

        # Summary card
        with st.container(border=True):
            st.markdown(f"**Age:** {st.session_state.age}")

            psa = current_psa()
            if psa is not None:
                try:
                    st.markdown(f"**Current PSA:** {float(psa):.2f} ng/mL")
                except Exception:
                    st.markdown(f"**Current PSA:** {psa} ng/mL")
            else:
                st.markdown("**Current PSA:** Not entered")

            risks = []
            if st.session_state.black_ancestry:
                risks.append("Black Race")
            if st.session_state.strong_fh:
                risks.append("Fam Hx")
            if st.session_state.germline_mutation:
                risks.append("Germline")
            st.markdown(f"**Risks:** {', '.join(risks) if risks else 'Avg'}")

        # PSA history timeline
        with st.container(border=True):
            st.markdown("**PSA history (date-stamped)**")
            hist = st.session_state.get("psa_history", [])
            if isinstance(hist, list) and len(hist) > 0:
                # ISO dates sort lexicographically; still guard for missing keys
                try:
                    hist_sorted = sorted(hist, key=lambda x: x.get("date", ""))
                except Exception:
                    hist_sorted = hist
                for e in hist_sorted:
                    d = e.get("date", "")
                    v = e.get("psa", "")
                    lbl = e.get("label", "PSA")
                    try:
                        v_str = f"{float(v):.2f}"
                    except Exception:
                        v_str = str(v)
                    st.write(f"{d} — **{v_str}** ng/mL ({lbl})")
            else:
                st.caption("No saved PSA values yet.")

            c1, c2 = st.columns(2)
            with c1:
                st.button("Save active PSA", on_click=_cb_save_active_psa, use_container_width=True)
            with c2:
                st.button("Clear PSA history", on_click=lambda: st.session_state.update({"psa_history": []}), use_container_width=True)

        # 2. Workup metrics (PSAD, %Free, MRI, biomarkers)
        psad = current_psad()
        pctfree = current_free_psa_pct()
        mri_done = bool(st.session_state.get("mri_done", False))
        pirads = st.session_state.get("pirads", "No MRI")
        bm_entries = st.session_state.get("biomarker_entries", [])

        if (psad is not None) or (pctfree is not None) or mri_done or (isinstance(bm_entries, list) and len(bm_entries) > 0):
            with st.container(border=True):
                st.markdown("**Pre-biopsy workup**")
                if psad is not None:
                    st.write(f"PSA density (PSAD): **{psad:.2f}** ng/mL/cc")
                if pctfree is not None:
                    st.write(f"% Free PSA: **{pctfree:.1f}%**")
                if mri_done:
                    st.write(f"MRI: **{pirads}**")

                # Biomarkers summary
                if isinstance(bm_entries, list) and bm_entries:
                    st.markdown("**Biomarkers**")
                    for e in bm_entries[-4:]:  # keep short
                        flag = "⚠️" if e.get("high_risk") else "✅"
                        st.write(f"{flag} {e.get('date','')} — {e.get('name','')}: {e.get('result','')}")

        # 3. Post-Biopsy Diagnosis

        if st.session_state.biopsy_result != "Not done":
            with st.container(border=True):
                bd = st.session_state.get('biopsy_date', None)
                if bd:
                    st.markdown(f"**Biopsy:** {bd} — {st.session_state.biopsy_result}")
                else:
                    st.markdown(f"**Biopsy:** {st.session_state.biopsy_result}")
                if st.session_state.risk_group:
                    color = "green" if "Low" in st.session_state.risk_group else "orange"
                    if "High" in st.session_state.risk_group or "Very" in st.session_state.risk_group:
                        color = "red"
                    st.markdown(f"**Group:** :{color}[{st.session_state.risk_group}]")

        st.divider()

        # 4. EXTERNAL CALCULATORS
        st.markdown("### 🌐 Pre-Biopsy Risk Calcs")
        st.caption("External tools to assess cancer probability:")
        st.link_button("PBCG Risk Calculator", "https://riskcalc.org/PBCG/")
        st.link_button("Rotterdam (ERSPC)", "https://www.prostatecancer-riskcalculator.com/")

        st.divider()
        st.button("🔄 Reset Patient", on_click=reset_all, use_container_width=True)

        st.markdown("---")
        st.caption(f"© {COPYRIGHT_YEAR} {APP_AUTHOR}")
        st.caption(f"Version {APP_VERSION}")


def progress_header() -> None:
    step = int(st.session_state.wizard_step)
    titles = {
        1: "Screening",
        2: "Risk refinement / pre-biopsy evaluation",
        3: "Biopsy result",
        4: "After negative biopsy (risk-based follow-up)",
        5: "Post-biopsy module"
    }
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;">
          <div style="font-size:34px;">🩺</div>
          <div>
            <div style="font-size:34px;font-weight:800;line-height:1.0;">Prostate Screening Navigator</div>
            <div style="color:#6b7280;margin-top:4px;">
              {APP_NAME} • v{APP_VERSION} • {APP_AUTHOR}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Step-by-step decision-support summary based on AUA/SUO early-detection concepts. "
        "Supports shared decision-making (SDM) and does not replace clinical judgment or local protocols."
    )

    st.markdown(f"### Step {step} of 5 — **{titles.get(step, '')}**")
    st.progress(step / 5.0)


def pathway_option(which: str) -> None:
    with st.expander("🗺️ Pathway (optional)"):
        st.caption("Highlighted node (blue) reflects where the app places the patient based on current inputs.")
        if which == "nb":
            st.graphviz_chart(dot_nb(active_node_nb()), use_container_width=True)
        else:
            st.graphviz_chart(dot_main(active_node_main()), use_container_width=True)
        zoom_url = f"?focus=1&path={which}"
        st.markdown(f"""<div style="margin-top:10px;"><a href="{zoom_url}" target="_blank" style="display:inline-block;padding:10px 12px;border:1px solid #d1d5db;border-radius:12px;text-decoration:none;">🔍 Zoom pathway (opens in new tab)</a></div>""", unsafe_allow_html=True)


# ============================================================
# STEP PAGES
# ============================================================
def page_screening() -> None:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Patient & screening inputs")
        st.caption('Required: age + life expectancy + risk factors. PSA/DRE optional but needed for downstream evaluation.')
        st.number_input("Age", min_value=40, max_value=90, step=1, key="age")
        st.checkbox("Life expectancy ≥ 10 years", key="life_expectancy_10y")

        st.markdown("#### Risk factors")
        c1, c2, c3 = st.columns(3)
        c1.checkbox("Black ancestry", key="black_ancestry")
        c2.checkbox("Germline (BRCA)", key="germline_mutation")
        c3.checkbox("Strong Fam Hx", key="strong_fh")

        st.markdown("#### PSA / DRE")
        st.checkbox("PSA performed", key="psa_done")
        if st.session_state.psa_done:
            c_psa, c_date, c_save = st.columns([1.0, 1.0, 0.9])
            with c_psa:
                st.number_input("PSA (ng/mL)", min_value=0.0, max_value=100.0, step=0.1, key="psa_value")
            with c_date:
                st.date_input("PSA date", key="psa_date")
            with c_save:
                st.button("Save PSA", on_click=_cb_save_initial_psa, use_container_width=True)
        st.checkbox("Abnormal / suspicious DRE", key="dre_abnormal")

        if st.session_state.dre_abnormal and not st.session_state.psa_done:
            st.warning("Abnormal DRE → **obtain/confirm PSA first**.")

    with right:
        st.subheader("Screening guidance (SDM)")
        rec, why = screening_recommendation()
        st.success(rec)
        st.caption(why)
        
        psa = current_psa()
        st.markdown("#### Quick context")
        st.write(f"- Risk tier: **{'Higher risk' if is_high_risk() else 'Average risk'}**")
        st.write(f"- Age-threshold used: **{psa_threshold(int(st.session_state.age)):.1f} ng/mL**")
        st.write(f"- Current PSA: **{'—' if psa is None else f'{psa:.2f}'}**")
        st.write(f"- Abnormal DRE: **{'Yes' if st.session_state.dre_abnormal else 'No'}**")

    pathway_option("main")
    st.divider()
    cols = st.columns([1, 1, 2])
    cols[0].button("🔄 Reset", on_click=reset_all, use_container_width=True)
    if cols[2].button("Next ➜ Evaluation", use_container_width=True):
        # Auto-save PSA into history when advancing
        if st.session_state.get("psa_done", False):
            _append_psa_history("PSA", st.session_state.get("psa_value"), st.session_state.get("psa_date"))
        set_step(2)


def page_evaluation() -> None:
    st.subheader("Risk refinement / pre-biopsy evaluation")
    st.caption('Required: PSA (and repeat PSA if newly elevated). Optional: MRI, PSAD, biomarkers (use SDM).')
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown("#### Confirm PSA first")
        if not st.session_state.psa_done:
            st.info("PSA is not entered. If DRE is abnormal, obtain PSA first.")
        else:
            st.checkbox("PSA repeated to confirm elevation", key="psa_repeated")
            if st.session_state.psa_repeated:
                c_rpsa, c_rdate, c_rsave = st.columns([1.0, 1.0, 0.9])
                with c_rpsa:
                    st.number_input("Repeat PSA (ng/mL)", min_value=0.0, max_value=100.0, step=0.1, key="psa_repeat_value")
                with c_rdate:
                    st.date_input("Repeat PSA date", key="psa_repeat_date")
                with c_rsave:
                    st.button("Save repeat PSA", on_click=_cb_save_repeat_psa, use_container_width=True)

        st.markdown("#### Advanced diagnostics (optional)")
        st.checkbox("MRI performed", key="mri_done")
        if st.session_state.mri_done:
            st.selectbox("PI-RADS", ["PI-RADS 1-2", "PI-RADS 3", "PI-RADS 4", "PI-RADS 5"], key="pirads")
        else:
            st.session_state.pirads = "No MRI"

        st.number_input("PSA density (ng/mL/cc)", min_value=0.0, max_value=2.0, step=0.01, key="psa_density")
        # Persist PSAD for later steps (e.g., after negative biopsy)
        try:
            psad_val = float(st.session_state.get("psa_density", 0.0) or 0.0)
        except Exception:
            psad_val = 0.0
        if psad_val > 0:
            st.session_state["psa_density_saved"] = psad_val
        elif float(st.session_state.get("calc_psa_density", 0.0) or 0.0) > 0:
            st.session_state["psa_density_saved"] = float(st.session_state.get("calc_psa_density", 0.0) or 0.0)
        biomarker_panel_ui('bm_pre')

    with right:
        with st.expander("🧮 Clinical Tools (PSAD, %Free)", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Density Calc**")
                st.number_input("Prostate Vol (cc)", 0.0, 200.0, key="prostate_vol")
                st.button("Calculate PSAD", on_click=_cb_calc_psad_from_inputs)
                if st.session_state.get("_psad_just_calculated"):
                    st.success(f"Density: {st.session_state.calc_psa_density:.2f}")
                    st.session_state["_psad_just_calculated"] = False
            with c2:
                st.markdown("**% Free PSA Calc**")
                st.number_input("Free PSA (ng/mL)", 0.0, 20.0, step=0.1, key="psa_free")
                st.date_input("Free PSA date", key="psa_free_date")
                cbtn1, cbtn2 = st.columns(2)
                with cbtn1:
                    if st.button("Calculate %", key="btn_calc_free", use_container_width=True):
                        psa = current_psa() or 0
                        res = calculate_free_psa_pct(float(st.session_state.get('psa_free', 0.0) or 0.0), float(psa))
                        st.session_state.calc_free_psa_pct = res
                        st.success(f"% Free: {res:.1f}%")
                with cbtn2:
                    st.button("Save Free PSA", on_click=_cb_save_free_psa, use_container_width=True)

        st.markdown("#### Next-step output")
        rec, why = evaluation_recommendation()
        st.success(rec)
        st.caption(why)
        st.markdown("**Note:** Check the Sidebar for External Risk Calculators (PBCG/Rotterdam) if considering biopsy.")

    pathway_option("main")
    st.divider()
    cols = st.columns([1, 1, 2])
    if cols[0].button("⬅️ Back", use_container_width=True): set_step(1)
    if cols[2].button("Next ➜ Biopsy result", use_container_width=True):
        # Auto-save repeat PSA (if used) when advancing
        if st.session_state.get("psa_done", False) and st.session_state.get("psa_repeated", False):
            _append_psa_history("Repeat PSA", st.session_state.get("psa_repeat_value"), st.session_state.get("psa_repeat_date"))
        set_step(3)


def page_biopsy_result() -> None:
    st.subheader("Biopsy result")
    st.caption('Required: record biopsy result to branch into follow-up or post-biopsy risk evaluation.')
    st.markdown("Enter result. **Negative** → Surveillance. **Positive** → Risk Stratification.")
    st.radio("Biopsy result", options=["Not done", "Negative", "Positive"], horizontal=True, key="biopsy_result")
    st.date_input("Biopsy date", key="biopsy_date")

    pathway_option("main")
    st.divider()
    cols = st.columns([1, 1, 2])
    if cols[0].button("⬅️ Back", use_container_width=True): set_step(2)

    if st.session_state.biopsy_result == "Negative":
        if cols[2].button("Next ➜ After negative biopsy", use_container_width=True): set_step(4)
    elif st.session_state.biopsy_result == "Positive":
        if cols[2].button("Next ➜ Risk Stratification", use_container_width=True):
            # Transfer PSA from Screening → Post-biopsy PSA field (risk__psa)
            # Do it RIGHT HERE at click time so it persists across reruns.
            try:
                _psa = _current_psa_from_screening()
                if _psa is not None:
                    st.session_state[rk('psa')] = float(_psa)
            except Exception:
                pass

            # Mark that we are entering the post-biopsy module from screening
            st.session_state["_enter_post_biopsy"] = True

            # Snapshot the Screening PSA *at the moment of handoff* (most reliable).
            # We'll apply it as the default on the post-biopsy PSA widget without
            # writing directly to the widget key (avoids Streamlit warning).
            try:
                st.session_state["_handoff_psa"] = _current_psa_from_screening()
            except Exception:
                st.session_state["_handoff_psa"] = None
            st.session_state["_handoff_psa_apply"] = True

            set_step(5)
    else:
        cols[2].button("Next (disabled)", disabled=True, use_container_width=True)


def page_after_negative_biopsy() -> None:
    st.subheader("After negative biopsy (risk-based follow-up)")
    st.caption('Optional: MRI, PSAD, biomarkers. Goal: risk-based decision on re-biopsy vs surveillance.')
    # Carry forward PSA density from prior step (if available)
    s = st.session_state
    try:
        saved_psad = s.get('psa_density_saved', None)
        calc_psad = float(s.get('calc_psa_density', 0.0) or 0.0)
        current_psad = float(s.get('psa_density', 0.0) or 0.0)
        default_psad = float(DEFAULTS.get('psa_density', 0.0) or 0.0)
        # If current PSAD is default/zero and we have a saved or calculated value, restore it
        if (current_psad == 0.0 or current_psad == default_psad):
            if saved_psad is not None:
                try:
                    saved_psad_f = float(saved_psad)
                    if saved_psad_f > 0:
                        s['psa_density'] = saved_psad_f
                        current_psad = saved_psad_f
                except Exception:
                    pass
            if (current_psad == 0.0 or current_psad == default_psad) and calc_psad > 0:
                s['psa_density'] = calc_psad
                s['psa_density_saved'] = calc_psad
    except Exception:
        pass
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("#### MRI / risk parameters")
        st.checkbox("MRI performed", key="mri_done")
        if st.session_state.mri_done:
            st.selectbox("PI-RADS", ["PI-RADS 1-2", "PI-RADS 3", "PI-RADS 4", "PI-RADS 5"], key="pirads")
        st.number_input("PSA density", min_value=0.0, max_value=2.0, step=0.01, key="psa_density")
        st.checkbox("Adjunct biomarkers (optional)", key="biomarker_done")
        if st.session_state.biomarker_done:
            biomarker_panel_ui('bm_nb')

    with right:
        rec, why = neg_biopsy_recommendation()
        st.success(rec)
        st.caption(why)
    pathway_option("nb")
    st.divider()
    # If the recommendation is to re-biopsy, route the user back to the Biopsy Result step
    rec_l = (rec or "").strip().lower()
    # Only route back when the recommended *next action* is to do a (repeat) biopsy
    need_biopsy = rec_l.startswith("repeat biopsy") or (rec_l.startswith("consider") and "biopsy" in rec_l and "mri" not in rec_l)
    cols = st.columns([1, 1, 2])
    if cols[0].button("⬅️ Back", use_container_width=True):
        set_step(3)
    if need_biopsy:
        if cols[2].button("Next ➜ Biopsy result", use_container_width=True):
            # Clear result so user can record the outcome of the (repeat) biopsy
            st.session_state.biopsy_result = "Not done"
            st.session_state.biopsy_date = datetime.date.today()
            set_step(3)
    else:
        if cols[2].button("Finish (back to Screening)", use_container_width=True):
            st.session_state.biopsy_result = "Not done"
            st.session_state.biopsy_date = datetime.date.today()
            set_step(1)

def page_risk_stratification():
    st.subheader("Step 5: NCCN Risk Stratification")
    st.info("Calculate the Post-Biopsy Risk Group.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Clinical T-Stage", ["cT1c", "cT2a", "cT2b", "cT2c", "cT3a", "cT3b"], key="clinical_t")

        with st.expander("Grade Group helper (from Gleason patterns)", expanded=False):
            st.caption("Pick the primary and secondary Gleason patterns to see the corresponding ISUP Grade Group.")
            p = st.selectbox("Primary Gleason pattern", [3, 4, 5], key="gg_primary")
            s = st.selectbox("Secondary Gleason pattern", [3, 4, 5], key="gg_secondary")
            tertiary5 = st.checkbox("Tertiary pattern 5 present", key="gg_tertiary5")
            gg = gleason_to_grade_group(p, s)
            if gg is not None:
                st.markdown(f"**Gleason {p}+{s}={p+s} → Grade Group {gg}**")
                if tertiary5:
                    st.caption("Note: Grade Group is based on primary+secondary patterns. A tertiary 5 can worsen prognosis; consider documenting it separately.")
                st.button("Use this Grade Group", use_container_width=True, on_click=_apply_grade_group_from_gleason)
            else:
                st.warning("Invalid Gleason pattern combination.")

        st.selectbox("Grade Group", [1, 2, 3, 4, 5], key="grade_group")
        st.slider("% Positive Cores", 0, 100, key="positive_cores")
        
    with c2:
        if st.button("Calculate Risk Group"):
            psa = current_psa() or 0
            risk = calculate_nccn_risk(st.session_state.clinical_t, st.session_state.grade_group, psa, st.session_state.positive_cores)
            st.session_state.risk_group = risk
            st.rerun()
            
        if st.session_state.risk_group:
            st.success(f"**Risk Group: {st.session_state.risk_group}**")
            
    st.divider()
    b_col, n_col = st.columns([1,5])
    if b_col.button("⬅ Back"): set_step(3)

# ============================================================


# ============================================================
# GLOBAL APP CONFIG
# ============================================================
PB_APP_NAME = "Prostate Risk Navigator"
PB_APP_VERSION = "2.6.1 (PSA history + Biomarkers + Pathway PDF)"
PB_APP_AUTHOR = "Mahziar Khazaali, MD"
PB_APP_YEAR = "2025"



# ============================================================
# PDF GENERATION LOGIC
# ============================================================
class ReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, f'{PB_APP_NAME} - Clinical Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf(text_content: str, dot_source: str | None) -> bytes:
    """Generates a PDF bytes object from the report text and optional DOT graph."""
    pdf = ReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=10)

    # 1. Clean Text for FPDF (Standard fonts don't support Unicode symbols)
    # Replacements for common symbols used in the app
    replacements = {
        "≥": ">=",
        "≤": "<=",
        "→": "->",
        "–": "-",
        "—": "--",
        "•": "-",
        "±": "+/-",
        "™": "(TM)"
    }
    clean_text = text_content
    for char, rep in replacements.items():
        clean_text = clean_text.replace(char, rep)
    
    # Ensure latin-1 compatibility (replace other unknown chars with ?)
    clean_text = clean_text.encode('latin-1', 'replace').decode('latin-1')

    # 2. Write Text
    pdf.multi_cell(0, 5, clean_text)
    
    # 3. Embed Graph (if available)
    if dot_source:
        try:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Pathway Diagram", 0, 1)
            
            # Render DOT to PNG using Graphviz
            src = graphviz.Source(dot_source)
            # 'pipe' returns the raw image data
            png_data = src.pipe(format='png')
            
            # Save to a temporary file because FPDF image() needs a file path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(png_data)
                tmp_path = tmp.name
            
            # Insert image (width=190mm fits standard A4/Letter margins)
            pdf.image(tmp_path, x=10, w=190)
        except Exception as e:
            pdf.set_font("Arial", 'I', 10)
            pdf.cell(0, 10, f"[Could not render graph image: {str(e)}]", 0, 1)

    # Output PDF to bytes string
    return pdf.output(dest='S').encode('latin-1')


# ============================================================
# STATIC NCCN REFERENCE CHARTS (DOT DATA)
# ============================================================
# (Kept identical to previous version for logic consistency)
STATIC_NCCN_CHARTS = {
    "PROS_3": """
    digraph PROS_3_LOW_RISK {
      rankdir=LR;
      graph [fontname="Arial", fontsize=10, splines=ortho, nodesep=0.6, ranksep=0.6];
      node  [fontname="Arial", fontsize=10, shape=box, style="filled", fillcolor="#F8F9FA"];
      edge  [fontname="Arial", fontsize=9];

      LE [shape=diamond, fillcolor="#FFD700", label="Expected\\nsurvival?"];
      GE10 [shape=box, label="≥10 y"];
      LT10 [shape=box, label="<10 y"];

      LE -> GE10;
      LE -> LT10;

      AS [label="Active surveillance\\n(preferred for most)"];
      RT [label="Radiation therapy (RT)"];
      RP [label="Radical prostatectomy (RP)"];

      GE10 -> AS;
      GE10 -> RT;
      GE10 -> RP;
      AS -> PROS2 [label="Progressive disease\\n→ Initial risk stratification\\n& staging workup\\n(PROS-2)"];

      RT -> BCR_RT [shape=diamond, label="Biochemical\\nrecurrence?"];
      BCR_RT -> PROS10 [label="Yes\\n→ RT recurrence\\n(PROS-10)"];
      BCR_RT -> PROS8A  [label="No\\n→ Monitoring\\n(PROS-8)"];

      RP -> PSA_RP [shape=diamond, label="PSA status?"];
      PSA_RP -> UNDET [label="Undetectable"];
      PSA_RP -> PERSIST [label="PSA persistence"];

      UNDET -> ADV [shape=diamond, label="Adverse features\\nor LN mets?"];
      ADV -> PROS8B [label="No\\n→ Monitoring\\n(PROS-8)"];
      ADV -> PROS9  [label="Yes\\n→ Monitoring (preferred)\\nor consider treatment\\n(PROS-9)"];
      PERSIST -> PROS9B [label="→ RP PSA persistence/\\nrecurrence (PROS-9)"];

      OBS [label="Observation"];
      LT10 -> OBS;
      OBS -> PROS8C [label="→ Monitoring\\n(PROS-8)"];
    }
    """,
    "PROS_4": """
    digraph PROS_4_FAV_INTERMEDIATE {
      rankdir=LR;
      graph [fontname="Arial", fontsize=10, splines=ortho, nodesep=0.6, ranksep=0.6];
      node  [fontname="Arial", fontsize=10, shape=box, style="filled", fillcolor="#F8F9FA"];
      edge  [fontname="Arial", fontsize=9];

      LE [shape=diamond, fillcolor="#FFD700", label="Expected\\nsurvival?"];
      GT10 [label=">10 y"];
      Y5_10 [label="5–10 y"];

      LE -> GT10;
      LE -> Y5_10;

      AS [label="Active surveillance\\n(selected patients)"];
      RP [label="Radical prostatectomy (RP)"];
      RT [label="Radiation therapy (RT)"];

      GT10 -> AS;
      GT10 -> RP;
      GT10 -> RT;
      AS -> PROS2 [label="Progressive disease\\n→ Initial risk stratification\\n& staging workup\\n(PROS-2)"];

      RP -> PSA_RP [shape=diamond, label="PSA status?"];
      PSA_RP -> UNDET [label="Undetectable"];
      PSA_RP -> PERSIST [label="PSA persistence"];
      UNDET -> ADV [shape=diamond, label="Adverse features\\nor LN mets?"];
      ADV -> PROS8A [label="No\\n→ Monitoring\\n(PROS-8)"];
      ADV -> PROS9  [label="Yes\\n→ Monitoring (preferred)\\nor consider treatment\\n(PROS-9)"];
      PERSIST -> PROS9B [label="→ RP PSA persistence/\\nrecurrence (PROS-9)"];
      RT -> BCR [shape=diamond, label="Biochemical\\nrecurrence?"];
      BCR -> PROS10 [label="Yes\\n→ RT recurrence\\n(PROS-10)"];
      BCR -> PROS8B [label="No\\n→ Monitoring\\n(PROS-8)"];
      # 5–10y branch
      OBS [label="Observation\\n(preferred)"];
      RT2 [label="RT"];
      Y5_10 -> RT2;
      Y5_10 -> OBS;
      OBS -> PROS8C [label="→ Monitoring\\n(PROS-8)"];
      RT2 -> PROS8D [label="→ Monitoring\\n(PROS-8)"];
    }
    """,
    "PROS_5": """
    digraph PROS_5_UNFAV_INTERMEDIATE {
      rankdir=LR;
      graph [fontname="Arial", fontsize=10, splines=ortho, nodesep=0.6, ranksep=0.6];
      node  [fontname="Arial", fontsize=10, shape=box, style="filled", fillcolor="#F8F9FA"];
      edge  [fontname="Arial", fontsize=9];

      LE [shape=diamond, fillcolor="#FFD700", label="Expected\\nsurvival?"];
      GT10 [label=">10 y"];
      Y5_10 [label="5–10 y"];

      LE -> GT10;
      LE -> Y5_10;

      RP [label="Radical prostatectomy (RP)"];
      RT_ADT [label="RT + ADT\\n(4–6 mo)"];
      GT10 -> RP;
      GT10 -> RT_ADT;

      RP -> PSA_RP [shape=diamond, label="PSA status?"];
      PSA_RP -> UNDET [label="Undetectable"];
      PSA_RP -> PERSIST [label="PSA persistence"];
      UNDET -> ADV [shape=diamond, label="Adverse features\\nor LN mets?"];
      ADV -> PROS8A [label="No\\n→ Monitoring\\n(PROS-8)"];
      ADV -> PROS9  [label="Yes\\n→ Monitoring (preferred)\\nor consider treatment\\n(PROS-9)"];
      PERSIST -> PROS9B [label="→ RP PSA persistence/\\nrecurrence (PROS-9)"];

      RT_ADT -> BCR [shape=diamond, label="Biochemical\\nrecurrence?"];
      BCR -> PROS10 [label="Yes\\n→ RT recurrence\\n(PROS-10)"];
      BCR -> PROS8B [label="No\\n→ Monitoring\\n(PROS-8)"];

      OBS [label="Observation"];
      Y5_10 -> OBS;
      OBS -> PROS8C [label="→ Monitoring\\n(PROS-8)"];
    }
    """,
    "PROS_6": """
    digraph PROS_6_HIGH_VERY_HIGH {
      rankdir=LR;
      graph [fontname="Arial", fontsize=10, splines=ortho, nodesep=0.6, ranksep=0.6];
      node  [fontname="Arial", fontsize=10, shape=box, style="filled", fillcolor="#F8F9FA"];
      edge  [fontname="Arial", fontsize=9];
      LE [shape=diamond, fillcolor="#FFD700", label="Expected survival\\n>5 y OR symptomatic?"];
      YES [label="Yes"];
      NO  [label="≤5 y and\\nasymptomatic"];
      LE -> YES;
      LE -> NO;
      RT_ADT [label="RT + ADT\\n(12–36 mo)\\nOR (very-high only)\\nRT+ADT (24 mo)\\n+ abiraterone"];
      RP [label="RP\\n(select patients)"];

      YES -> RT_ADT;
      YES -> RP;
      RT_ADT -> BCR [shape=diamond, label="Biochemical\\nrecurrence?"];
      BCR -> PROS10 [label="Yes\\n→ RT recurrence\\n(PROS-10)"];
      BCR -> PROS8A [label="No\\n→ Monitoring\\n(PROS-8)"];
      RP -> PSA_RP [shape=diamond, label="PSA status?"];
      PSA_RP -> UNDET [label="Undetectable"];
      PSA_RP -> PERSIST [label="PSA persistence"];
      UNDET -> ADV [shape=diamond, label="Adverse features\\nor LN mets?"];
      ADV -> PROS8B [label="No\\n→ Monitoring\\n(PROS-8)"];
      ADV -> PROS9  [label="Yes\\n→ Monitoring (preferred)\\nor consider treatment\\n(PROS-9)"];
      PERSIST -> PROS9B [label="→ RP PSA persistence/\\nrecurrence (PROS-9)"];

      OBS_OR_RT [label="Observation\\nor RT+ADT"];
      NO -> OBS_OR_RT;
      OBS_OR_RT -> SYMP [shape=diamond, label="Symptomatic\\nprogression"];
      SYMP -> PROS13 [label="→ Metachronous\\noligometastatic M1 CSPC\\n(PROS-13)"];
      SYMP -> PROS14 [label="→ Low-volume M1 CSPC\\n(PROS-14)"];
      SYMP -> PROS15 [label="→ High-volume M1 CSPC\\n(PROS-15)"];
    }
    """,
    "PROS_7": """
    digraph PROS_7_REGIONAL_N1_M0 {
      rankdir=LR;
      graph [fontname="Arial", fontsize=10, splines=ortho, nodesep=0.6, ranksep=0.6];
      node  [fontname="Arial", fontsize=10, shape=box, style="filled", fillcolor="#F8F9FA"];
      edge  [fontname="Arial", fontsize=9];
      LE [shape=diamond, fillcolor="#FFD700", label="Expected survival\\n>5 y OR symptomatic?"];
      YES [label="Yes"];
      NO  [label="≤5 y and\\nasymptomatic"];
      LE -> YES;
      LE -> NO;
      RT_ADT_ABI [label="RT + ADT (24 mo)\\n+ abiraterone (preferred)\\nOR RT+ADT (24–36 mo)"];
      ADT_ABI [label="ADT + abiraterone"];
      RP [label="RP\\n(select patients)"];
      YES -> RT_ADT_ABI;
      YES -> ADT_ABI;
      YES -> RP;

      RT_ADT_ABI -> BCR [shape=diamond, label="Biochemical\\nrecurrence?"];
      BCR -> PROS10 [label="Yes\\n→ RT recurrence\\n(PROS-10)"];
      BCR -> PROS8A [label="No\\n→ Monitoring\\n(PROS-8)"];

      ADT_ABI -> PROS8B [label="→ Monitoring\\n(PROS-8)"];

      RP -> PSA_RP [shape=diamond, label="PSA status?"];
      PSA_RP -> UNDET [label="Undetectable"];
      PSA_RP -> PERSIST [label="PSA persistence"];
      UNDET -> ADV [shape=diamond, label="Adverse features\\nor LN mets?"];
      ADV -> PROS8C [label="No\\n→ Monitoring\\n(PROS-8)"];
      ADV -> PROS9  [label="Yes\\n→ Monitoring / consider treatment\\n(PROS-9)"];
      PERSIST -> PROS9B [label="→ RP PSA persistence/\\nrecurrence (PROS-9)"];
      # ≤5y asymptomatic
      LOWLE [label="Observation\\nor RT\\nor ADT ± RT"];
      NO -> LOWLE;
      LOWLE -> SYMP [shape=diamond, label="Symptomatic\\nprogression"];
      SYMP -> PROS13 [label="→ PROS-13"];
      SYMP -> PROS14 [label="→ PROS-14"];
      SYMP -> PROS15 [label="→ PROS-15"];
      SYMP -> PROS17 [label="→ M1 CRPC\\n(PROS-17)"];
    }
    """,
    "PROS_8": """
    digraph PROS_8_MONITORING_RECURRENCE {
      rankdir=LR;
      graph [fontname="Arial", fontsize=10, splines=ortho, nodesep=0.6, ranksep=0.6];
      node  [fontname="Arial", fontsize=10, shape=box, style="filled", fillcolor="#F8F9FA"];
      edge  [fontname="Arial", fontsize=9];
      INIT [label="After initial definitive therapy\\nMonitoring:\\nPSA q6–12 mo x5y,\\nthen annually;\\nconsider DRE if suspicion"];
      INIT -> POST_RP;
      INIT -> POST_RT;
      INIT -> RADIO;

      POST_RP [shape=diamond, label="Post-RP:\\nPSA persistence/\\nrecurrence?"];
      POST_RP -> PROS9 [label="Yes\\n→ PROS-9"];

      POST_RT [shape=diamond, label="Post-RT:\\nPSA recurrence\\nor +DRE?"];
      POST_RT -> PROS10 [label="Yes\\n→ PROS-10"];

      RADIO [shape=diamond, label="Radiographic mets\\nwithout PSA recurrence?"];
      RADIO -> BIOPSY [label="Yes\\nBiopsy mets site"];
      BIOPSY -> PROS13 [label="→ PROS-13"];
      BIOPSY -> PROS14 [label="→ PROS-14"];
      BIOPSY -> PROS15 [label="→ PROS-15"];
      N1 [label="N1 on ADT OR\\nLocalized on observation"];
      N1 -> FUP [label="PE + PSA q3–6 mo\\nImaging for symptoms\\nor ↑PSA"];
      FUP [shape=diamond, label="Progression?"];
      FUP -> M0 [label="N1, M0"];
      FUP -> M1 [label="M1"];
      M0 -> PROS16 [label="→ Systemic therapy\\nfor M0 CRPC\\n(PROS-16)"];
      M1 -> PROS17 [label="→ Workup & treatment\\nof M1 CRPC\\n(PROS-17)"];
    }
    """,
    "PROS_14": """
    digraph PROS_14_LOW_VOLUME_M1_CSPC {
      rankdir=LR;
      graph [fontname="Arial", fontsize=10, splines=ortho, nodesep=0.6, ranksep=0.6];
      node  [fontname="Arial", fontsize=10, shape=box, style="filled", fillcolor="#F8F9FA"];
      edge  [fontname="Arial", fontsize=9];
      WORKUP [label="Workup:\\nPE; imaging for staging;\\nPSA/PSADT;\\nestimate life expectancy;\\nconsider germline/somatic testing;\\nQOL measures"];

      TYPE [shape=diamond, label="Presentation"];
      WORKUP -> TYPE;
      SYNCH [label="Synchronous low-volume\\nor synchronous oligometastatic"];
      META  [label="Metachronous low-volume"];
      TYPE -> SYNCH;
      TYPE -> META;
      TX1 [label="ADT + one of:\\nPreferred: abiraterone / apalutamide / enzalutamide\\nOther: darolutamide\\nOR ADT+docetaxel + one of above\\nOR ADT + EBRT to primary tumor\\n(+/- systemic) (selected)"];
      SYNCH -> TX1;

      TX2 [label="ADT + one of:\\nPreferred: abiraterone / apalutamide / enzalutamide\\nOther: darolutamide"];
      META -> TX2;
      MON [label="Follow-up:\\nPE + PSA q3–6 mo\\nImaging for symptoms\\nPeriodic imaging to monitor response"];
      TX1 -> MON;
      TX2 -> MON;
      PROG [shape=diamond, label="Progression on\\nADT + ARPI?"];
      MON -> PROG;
      PROG -> PROS17 [label="Yes\\n→ M1 CRPC\\n(PROS-17)"];
    }
    """,
    "PROS_15": """
    digraph PROS_15_HIGH_VOLUME_M1_CSPC {
      rankdir=LR;
      graph [fontname="Arial", fontsize=10, splines=ortho, nodesep=0.6, ranksep=0.6];
      node  [fontname="Arial", fontsize=10, shape=box, style="filled", fillcolor="#F8F9FA"];
      edge  [fontname="Arial", fontsize=9];
      WORKUP [label="Workup:\\nPE; imaging for staging;\\nPSA/PSADT;\\nestimate life expectancy;\\nconsider germline/somatic testing;\\nQOL measures"];

      TYPE [shape=diamond, label="Synchronous or\\nmetachronous\\nhigh-volume mets"];
      WORKUP -> TYPE;
      TX [label="ADT with one of:\\nADT + docetaxel + (abiraterone or darolutamide)\\nOR ADT + (abiraterone / apalutamide / enzalutamide)\\n(see guideline specifics)"];
      TYPE -> TX;

      MON [label="Follow-up:\\nPE + PSA q3–6 mo\\nImaging for symptoms\\nPeriodic imaging to monitor response"];
      TX -> MON;
      PROG [shape=diamond, label="Progression"];
      MON -> PROG;
      PROG -> PROS17 [label="→ Workup & treatment\\nof M1 CRPC\\n(PROS-17)"];
    }
    """
}


def link_button_or_markdown(label: str, url: str):
    if hasattr(st, "link_button"):
        st.link_button(label, url)
    else:
        st.markdown(f"[{label}]({url})")


# ============================================================
# SHARED CSS
# ============================================================
st.markdown(
    """
    <style>
      /* Main Background and Font */
      .main { background-color: #fcfcfc; }
      
      /* Muted Text */
      .muted { color: rgba(49, 51, 63, 0.70); }
      .tiny { font-size: 0.90rem; }

      /* Cards for UI Elements */
      .card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      }
      .title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 5px;
        color: #2c3e50;
      }
      .subtitle {
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-bottom: 12px;
      }
      
      /* Badges and Tags */
      .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        margin-right: 5px;
        background-color: #f0f2f6;
        color: #31333f;
        border: 1px solid #d1d5db;
        font-weight: 600;
      }
      
      /* Step Counter */
      .step {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        background-color: #e8f4f8;
        color: #0068c9;
        font-weight: 700;
        border: 1px solid #bce3f2;
      }

      /* Life Expectancy Result Box */
      .le-result-title { font-size: 1.10rem; font-weight: 700; margin-bottom: 0.15rem; }
      .le-result-sub { font-size: 0.90rem; color: rgba(49, 51, 63, 0.75); margin-bottom: 0.35rem; }
      .val-unit { font-size: 0.90rem; color: rgba(49, 51, 63, 0.75); margin-bottom: 0.15rem; }
      .val-number { font-size: 2.6rem; font-weight: 800; line-height: 1.0; }
      .val-number-red { color: #d9534f; font-weight: 900; }
      .val-wrap { margin-top: 0.15rem; }

      /* Tables from Review Assets */
      .pn-table {
        width: 100%;
        border-collapse: collapse;
        border-spacing: 0;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
      }
      .pn-table th {
        background: #f8fafc;
        text-align: left;
        padding: 12px 16px;
        border-bottom: 2px solid #e5e7eb;
        font-weight: 600;
        color: #1e293b;
      }
      .pn-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #f1f5f9;
        color: #334155;
      }
      .pn-table tr:last-child td { border-bottom: none; }
      .pn-table ul { margin: 0; padding-left: 1.2rem; }
      .pn-table li { margin: 0.2rem 0; }

      .pn-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        background: #f1f5f9;
        font-size: 0.8rem;
        margin-top: 6px;
      }
      .pn-section-row td {
        background: #f8fafc;
        font-weight: 600;
        border-bottom: 1px solid #e2e8f0;
      }
      
      /* Info Card Styles */
      .pn-card {
        border-left: 4px solid #3b82f6;
        background: #eff6ff;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 14px;
      }
      
      .pn-muted { color: #64748b; font-size: 0.9rem; }
      
    </style>
    """,
    unsafe_allow_html=True,
)


def render_value_block(unit: str, value_str: str, red: bool = False) -> None:
    cls = "val-number val-number-red" if red else "val-number"
    st.markdown(
        f"""
        <div class="val-wrap">
          <div class="val-unit">{unit}</div>
          <div class="{cls}">{value_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def ul(items: list[str]) -> str:
    return "<ul>" + "".join([f"<li>{x}</li>" for x in items]) + "</ul>"

def table_html(headers: list[str], rows: list[list[str]]) -> str:
    thead = "<thead><tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr></thead>"
    tbody = "<tbody>" + "".join(
        [
            "<tr>" + "".join([f"<td>{cell}</td>" for cell in row]) + "</tr>"
            for row in rows
        ]
    ) + "</tbody>"
    return f"<table class='pn-table'>{thead}{tbody}</table>"


# ============================================================
# WORKFLOW NAV (Cleaned up buttons)
# ============================================================
def _ensure_workflow_state():
    if "workflow_step" not in st.session_state:
        st.session_state["workflow_step"] = 1  # 1..4


def goto_step(n: int):
    st.session_state["workflow_step"] = int(n)
    st.rerun()


def workflow_nav_buttons(
    *,
    back_step: int | None,
    next_step: int | None,
    back_disabled: bool = False,
    next_disabled: bool = False,
    next_label: str = "Next Step ➡",
    back_label: str = "⬅ Back",
    extra_right_button: Tuple[str, int] | None = None,  # ("Open report", 4)
):
    st.divider()
    c1, c2, c3, c4 = st.columns([1, 1, 3, 1.5])
    with c1:
        if back_step is not None:
            if st.button(back_label, disabled=back_disabled, key=f"nav_back_{back_step}_{st.session_state.get('workflow_step')}"):
                goto_step(back_step)
    with c2:
        if next_step is not None:
            if st.button(next_label, disabled=next_disabled, key=f"nav_next_{next_step}_{st.session_state.get('workflow_step')}", type="primary"):
                goto_step(next_step)
    with c4:
        if extra_right_button:
            lbl, step_n = extra_right_button
            if st.button(lbl, key=f"nav_extra_{step_n}_{st.session_state.get('workflow_step')}"):
                goto_step(step_n)


# ============================================================
# 1) RISK STRATIFICATION ENGINE
# ============================================================
DEFAULT_CT_STAGE = "cT1c"
DEFAULT_CN_STAGE = "cN0"
DEFAULT_CM_STAGE = "cM0"

DEFAULT_PT_STAGE = "pT2"
DEFAULT_PN_STAGE = "pN0"

DEFAULT_PSA = 6.5
DEFAULT_PRIMARY_PATTERN = 3
DEFAULT_SECONDARY_PATTERN = 3
DEFAULT_CORE_PERCENT = 10

DEFAULT_MARGIN = "Unknown / not applicable"
MARGIN_OPTIONS = [
    DEFAULT_MARGIN,
    "Negative margin (R0)",
    "Focally positive margin (R1, focal)",
    "Extensively positive margin (R1, extensive)",
]
MARGIN_EXPLANATIONS = {
    DEFAULT_MARGIN: "Margin status not available or not applicable (no radical prostatectomy data entered).",
    "Negative margin (R0)": "No tumor at the inked surgical margin.",
    "Focally positive margin (R1, focal)": "Limited tumor at the inked margin; recurrence risk depends on location and length.",
    "Extensively positive margin (R1, extensive)": "Extensive tumor at the inked margin; associated with higher risk of biochemical recurrence.",
}

CORE_SITES = [
    ("A", "LEFT LATERAL BASE"),
    ("B", "LEFT LATERAL MID"),
    ("C", "LEFT LATERAL APEX"),
    ("D", "LEFT MEDIAL BASE"),
    ("E", "LEFT MEDIAL MID"),
    ("F", "LEFT MEDIAL APEX"),
    ("G", "RIGHT MEDIAL BASE"),
    ("H", "RIGHT MEDIAL MID"),
    ("I", "RIGHT MEDIAL APEX"),
    ("J", "RIGHT LATERAL BASE"),
    ("K", "RIGHT LATERAL MID"),
    ("L", "RIGHT LATERAL APEX"),
]
TOTAL_SYSTEMATIC_CORES = len(CORE_SITES)

TARGETED_SITES = [
    ("T1", "Targeted core 1"),
    ("T2", "Targeted core 2"),
    ("T3", "Targeted core 3"),
]

T_STAGE_DEFINITIONS = {
    "cTX": "Primary tumor cannot be assessed",
    "cT0": "No evidence of primary tumor",
    "cT1a": "Tumor incidental histologic finding in 5% or less of tissue resected",
    "cT1b": "Tumor incidental histologic finding in more than 5% of tissue resected",
    "cT1c": "Tumor identified by needle biopsy found in one or both sides, but not palpable",
    "cT2a": "Tumor is palpable and confined within prostate; involves one-half of one side or less",
    "cT2b": "Tumor is palpable and confined within prostate; involves more than one-half of one side but not both sides",
    "cT2c": "Tumor is palpable and confined within prostate; involves both sides",
    "cT3a": "Extraprostatic tumor that is not fixed and does not invade adjacent structures",
    "cT3b": "Tumor invades seminal vesicle(s)",
    "cT4": "Tumor is fixed or invades adjacent structures other than seminal vesicles such as external sphincter, rectum, bladder, levator muscles, and/or pelvic wall",
}
T_STAGE_OPTIONS = list(T_STAGE_DEFINITIONS.keys())

N_STAGE_DEFINITIONS = {
    "cNX": "Regional lymph nodes cannot be assessed",
    "cN0": "No regional lymph node metastasis",
    "cN1": "Metastasis in regional lymph node(s)",
}
N_STAGE_OPTIONS = list(N_STAGE_DEFINITIONS.keys())

M_STAGE_DEFINITIONS = {
    "cM0": "No distant metastasis",
    "cM1": "Distant metastasis present",
    "cM1a": "Nonregional lymph node(s)",
    "cM1b": "Bone(s)",
    "cM1c": "Other site(s) with or without bone disease",
}
M_STAGE_OPTIONS = list(M_STAGE_DEFINITIONS.keys())

PT_STAGE_DEFINITIONS = {
    "pT2": "Organ confined (including apex and capsule involvement, but not beyond prostate)",
    "pT3a": "Extraprostatic extension (unilateral or bilateral) and/or microscopic bladder neck invasion",
    "pT3b": "Tumor invades seminal vesicle(s)",
    "pT4": "Tumor is fixed or invades structures other than seminal vesicles (external sphincter, rectum, bladder, levator muscles, and/or pelvic wall)",
}
PT_STAGE_OPTIONS = list(PT_STAGE_DEFINITIONS.keys())

PN_STAGE_DEFINITIONS = {
    "pNX": "Regional lymph nodes cannot be assessed pathologically",
    "pN0": "No positive regional lymph nodes on pathology",
    "pN1": "Metastases in regional lymph node(s) on pathology",
}
PN_STAGE_OPTIONS = list(PN_STAGE_DEFINITIONS.keys())


def _normalize_stage(stage: str, letter: str) -> str:
    s = stage.strip().lower()
    if s.startswith(("c", "p")):
        s = s[1:]
    if not s.startswith(letter):
        raise ValueError(f"{letter.upper()} stage must start with {letter.upper()}, c{letter.upper()}, or p{letter.upper()}")
    return s


def normalize_t_stage(t_stage: str) -> str:
    return _normalize_stage(t_stage, "t")


def normalize_n_stage(n_stage: str) -> str:
    return _normalize_stage(n_stage, "n")


def normalize_m_stage(m_stage: str) -> str:
    return _normalize_stage(m_stage, "m")


def is_t1(t_stage: str) -> bool:
    return normalize_t_stage(t_stage).startswith("t1")


def is_t2a(t_stage: str) -> bool:
    return normalize_t_stage(t_stage).startswith("t2a")


def is_t2b_to_t2c(t_stage: str) -> bool:
    s = normalize_t_stage(t_stage)
    return s.startswith("t2b") or s.startswith("t2c")


def is_t3_to_t4(t_stage: str) -> bool:
    s = normalize_t_stage(t_stage)
    return s.startswith("t3") or s.startswith("t4")


def has_regional_nodes(n_stage: str) -> bool:
    return normalize_n_stage(n_stage).startswith("n1")


def has_distant_metastasis(m_stage: str) -> bool:
    return normalize_m_stage(m_stage).startswith("m1")


def gleason_to_grade_group(primary: int, secondary: int):
    score = primary + secondary
    pattern = f"{primary}+{secondary}"
    if primary == 3 and secondary == 3:
        return 1, score, f"Gleason {pattern}={score} → Grade Group 1"
    elif primary == 3 and secondary == 4:
        return 2, score, f"Gleason {pattern}={score} → Grade Group 2"
    elif primary == 4 and secondary == 3:
        return 3, score, f"Gleason {pattern}={score} → Grade Group 3"
    elif (primary == 4 and secondary == 4) or (primary == 3 and secondary == 5) or (primary == 5 and secondary == 3):
        return 4, score, f"Gleason {pattern}={score} → Grade Group 4"
    elif (primary == 4 and secondary == 5) or (primary == 5 and secondary == 4) or (primary == 5 and secondary == 5):
        return 5, score, f"Gleason {pattern}={score} → Grade Group 5"
    else:
        return None, score, f"Gleason {pattern}={score} → pattern not in standard Grade Group table"


def classify_nccn_risk(
    t_stage: str,
    n_stage: str,
    m_stage: str,
    grade_group: int,
    psa: float,
    cores_positive: int,
    cores_total: int,
):
    details = []
    details.append(f"Clinical TNM: {t_stage}, {n_stage}, {m_stage}")
    details.append(f"PSA {psa:.1f} ng/mL, highest biopsy Grade Group {grade_group}")

    if cores_total <= 0:
        raise ValueError("Total cores must be > 0")

    percent_cores = 100.0 * cores_positive / cores_total
    details.append(f"Systematic cores positive for cancer: {cores_positive}/{cores_total} ({percent_cores:.1f}%)")

    if has_regional_nodes(n_stage) or has_distant_metastasis(m_stage):
        details.append("N1 and/or M1 present → outside clinically localized NCCN risk groups.")
        return "Metastatic/regional disease (outside localized risk groups)", None, details

    very_high_flags = [is_t3_to_t4(t_stage), grade_group in (4, 5), psa > 40.0]
    if sum(very_high_flags) >= 2:
        details.append("≥2 of: T3–T4, Grade Group 4–5, PSA >40 → Very high-risk.")
        return "Very high", None, details

    high_flag = is_t3_to_t4(t_stage) or (grade_group in (4, 5)) or (psa > 20.0)
    if high_flag:
        details.append("At least one of: T3–T4, Grade Group 4–5, PSA >20 → High-risk.")
        return "High", None, details

    irfs = []
    if is_t2b_to_t2c(t_stage):
        irfs.append("T2b–T2c")
    if grade_group in (2, 3):
        irfs.append("Grade Group 2–3")
    if 10.0 <= psa <= 20.0:
        irfs.append("PSA 10–20")

    if irfs:
        details.append(f"Intermediate-risk factors: {', '.join(irfs)}")
        if (len(irfs) == 1) and (grade_group in (1, 2)) and (percent_cores < 50.0):
            details.append("Favorable intermediate: 1 IRF, GG 1–2, <50% cores positive.")
            return "Intermediate", "Favorable", details
        else:
            details.append("Unfavorable intermediate: ≥2 IRFs and/or GG 3 and/or ≥50% cores positive.")
            return "Intermediate", "Unfavorable", details

    if (is_t1(t_stage) or is_t2a(t_stage)) and grade_group == 1 and psa < 10.0:
        details.append("T1–T2a, Grade Group 1, PSA <10 → Low-risk.")
        return "Low", None, details

    details.append("Does not meet NCCN low/intermediate/high/very-high criteria with given data.")
    return "Unclassifiable", None, details


def get_additional_evaluation_recommendations(risk_group: str | None, subcategory: str | None):
    if risk_group is None:
        return None, []
    rg = risk_group.strip()
    sub = (subcategory or "").strip()

    if rg == "Low":
        return "Low-risk localized prostate cancer", [
            "If active surveillance is being considered, confirm that the patient is an appropriate candidate.",
            "Consider confirmatory testing such as repeat biopsy, multiparametric MRI, and/or validated biomarkers.",
        ]

    if rg == "Intermediate" and sub == "Favorable":
        return "Favorable intermediate-risk localized prostate cancer", [
            "If active surveillance or conservative treatment is being considered, confirm that criteria are met.",
            "Consider confirmatory testing (repeat biopsy, mpMRI, and/or biomarkers) to refine risk assessment.",
        ]

    if rg == "Intermediate" and sub == "Unfavorable":
        return "Unfavorable intermediate-risk localized prostate cancer", [
            "Obtain pelvic soft-tissue imaging (CT or MRI).",
            "Consider bone imaging (bone scan or PSMA PET/CT), especially with higher PSA or multiple risk factors.",
            "If regional metastases are found, follow node-positive (cN1) pathways.",
            "If distant metastases are found, use metastatic castration-sensitive prostate cancer pathways.",
        ]

    if rg == "High":
        return "High-risk localized prostate cancer", [
            "Obtain pelvic CT or MRI.",
            "Obtain bone imaging (bone scan or PSMA PET/CT).",
            "If regional lymph node metastases are detected, manage as node-positive disease.",
            "If distant metastases are detected, manage as metastatic castration-sensitive prostate cancer.",
        ]

    if rg == "Very high":
        return "Very high-risk localized prostate cancer", [
            "Obtain pelvic CT or MRI.",
            "Obtain bone imaging (bone scan or PSMA PET/CT); there is a high probability of metastatic disease.",
            "If regional lymph node metastases are detected, manage as node-positive disease.",
            "If distant metastases are detected, follow metastatic castration-sensitive pathways and consider clinical trials.",
        ]

    if "Metastatic/regional" in rg:
        return "Regional or metastatic disease", [
            "Ensure complete staging with cross-sectional imaging and bone imaging (or PSMA PET/CT).",
            "Classify disease volume/distribution (node-only vs bone vs visceral) to guide therapy choices.",
        ]

    return None, []


def classify_disease_category(n_stage: str, m_stage: str) -> str:
    if has_distant_metastasis(m_stage):
        return "Metastatic CSPC (M1)"
    if has_regional_nodes(n_stage) and not has_distant_metastasis(m_stage):
        return "Node-positive (N1, M0)"
    if (not has_regional_nodes(n_stage)) and not has_distant_metastasis(m_stage):
        return "Localized"
    return "Uncertain"


def get_treatment_options(disease_category: str, risk_group: str | None, subcategory: str | None):
    sections: list[tuple[str, list[str]]] = []

    if disease_category.startswith("Metastatic"):
        title = "Metastatic castration-sensitive prostate cancer (M1)"
        sections.append(
            (
                "Systemic backbone",
                [
                    "Androgen deprivation therapy (ADT) is the backbone for all patients.",
                    "ADT monotherapy is generally avoided in fit patients; intensification improves survival.",
                ],
            )
        )
        sections.append(
            (
                "Intensification options (examples)",
                [
                    "ADT + a next-generation AR-targeted agent (e.g., abiraterone, apalutamide, enzalutamide, darolutamide), considering comorbidities/interactions.",
                    "ADT + docetaxel in selected patients (often higher-volume disease).",
                    "ADT + docetaxel + AR-targeted agent in very fit patients according to contemporary trials.",
                ],
            )
        )
        sections.append(
            (
                "Additional considerations",
                [
                    "Stratify by disease volume (low vs high; oligometastatic vs polymetastatic).",
                    "Consider prostate-directed RT in selected low-volume metastatic cases.",
                    "Discuss clinical trials when available.",
                ],
            )
        )
        return title, sections

    if disease_category.startswith("Node-positive"):
        title = "Clinically node-positive, non-metastatic prostate cancer (N1, M0)"
        sections.append(
            (
                "Typical management frameworks",
                [
                    "Definitive radiotherapy to the prostate ± pelvic nodes plus long-term ADT.",
                    "Systemic intensification (e.g., addition of an AR-targeted agent) may be appropriate; check current guidelines.",
                ],
            )
        )
        sections.append(
            (
                "Other options / special situations",
                [
                    "Selected patients may undergo radical prostatectomy with extended lymph node dissection in high-volume centers, usually within a multimodal plan.",
                    "Consider multidisciplinary tumor board discussion and clinical trials.",
                ],
            )
        )
        sections.append(
            (
                "Supportive / follow-up care",
                [
                    "Monitor PSA and testosterone to confirm castration levels on ADT.",
                    "Address bone health, cardiovascular risk, and quality-of-life issues from long-term ADT.",
                ],
            )
        )
        return title, sections

    if disease_category == "Localized" and risk_group:
        rg = risk_group.strip()
        sub = (subcategory or "").strip()
        title = f"Treatment options for localized {rg.lower()} risk prostate cancer"

        if rg == "Low":
            sections.append(
                (
                    "Conservative / surveillance",
                    [
                        "Active surveillance is appropriate for many patients with low-risk disease and adequate life expectancy.",
                        "Observation/watchful waiting may be considered with limited life expectancy or major comorbidities (individualize).",
                    ],
                )
            )
            sections.append(
                (
                    "Definitive local therapy",
                    [
                        "Radical prostatectomy with pelvic lymph node assessment in selected patients.",
                        "External beam radiotherapy (EBRT) with modern dose-escalated techniques.",
                        "Brachytherapy in centers with expertise.",
                    ],
                )
            )
            return title, sections

        if rg == "Intermediate" and sub == "Favorable":
            sections.append(
                (
                    "Conservative options (selected patients)",
                    [
                        "Active surveillance may be considered in carefully selected favorable intermediate-risk men (limited volume, GG 1–2, low PSA density).",
                        "Observation may be reasonable with limited life expectancy (individualize).",
                    ],
                )
            )
            sections.append(
                (
                    "Definitive local therapy",
                    [
                        "Radical prostatectomy ± pelvic lymph node dissection.",
                        "EBRT alone, or EBRT with short-term ADT in selected patients.",
                        "Brachytherapy-based approaches in experienced centers.",
                    ],
                )
            )
            return title, sections

        if rg == "Intermediate" and sub == "Unfavorable":
            sections.append(
                (
                    "Definitive local therapy (typical)",
                    [
                        "Radical prostatectomy with pelvic lymph node dissection as part of a multimodal plan.",
                        "EBRT + short-term ADT (e.g., 4–6 months).",
                        "EBRT + brachytherapy boost + ADT in selected, fit patients.",
                    ],
                )
            )
            sections.append(
                (
                    "Other considerations",
                    [
                        "Discuss risks/benefits of combined-modality vs single-modality treatment.",
                        "Consider clinical trials in appropriate candidates.",
                    ],
                )
            )
            return title, sections

        if rg in ("High", "Very high"):
            sections.append(
                (
                    "Multimodal therapy",
                    [
                        "EBRT to prostate/pelvis + long-term ADT (e.g., 18–36 months) is a standard backbone.",
                        "For very-high-risk, some guideline versions include EBRT + long-term ADT + abiraterone in selected patients.",
                        "Systemic intensification (e.g., docetaxel or an AR-targeted agent) may be considered in selected patients.",
                    ],
                )
            )
            sections.append(
                (
                    "Surgical options",
                    [
                        "Radical prostatectomy with extended pelvic lymph node dissection in carefully selected men, usually within a multimodal strategy.",
                    ],
                )
            )
            sections.append(
                (
                    "Additional points",
                    [
                        "Consider clinical trials for high and very high-risk disease.",
                        "Discuss potential need for adjuvant or early-salvage radiotherapy after surgery.",
                    ],
                )
            )
            return title, sections

    return None, []


def classify_ajcc_stage(t_stage: str, n_stage: str, m_stage: str, psa: float, grade_group: int | None):
    details: list[str] = []
    details.append(f"TNM used for staging: {t_stage}, {n_stage}, {m_stage}")
    details.append(f"PSA {psa:.1f} ng/mL")

    if grade_group is None:
        details.append("Grade Group not provided; AJCC 8th prognostic grouping requires Grade Group.")
        return "Stage group cannot be determined (Grade Group unknown)", details

    details.append(f"Grade Group {grade_group}")

    if has_distant_metastasis(m_stage):
        details.append("Any T, any N, M1 → Stage IVB.")
        return "Stage IVB", details

    if has_regional_nodes(n_stage):
        details.append("Any T, N1, M0 → Stage IVA.")
        return "Stage IVA", details

    sT = normalize_t_stage(t_stage)
    if sT in ("tx", "t0"):
        details.append("Primary tumor TX/T0 → Stage group cannot be assigned.")
        return "Stage cannot be determined (TX/T0)", details

    is_T1 = sT.startswith("t1")
    is_T2 = sT.startswith("t2")
    is_T2a_ = sT.startswith("t2a")
    is_T2b_c = sT.startswith("t2b") or sT.startswith("t2c")
    is_T3_4 = sT.startswith("t3") or sT.startswith("t4")

    if grade_group == 5:
        details.append("Any T, N0/NX, M0, Grade Group 5 → Stage IIIC.")
        return "Stage IIIC", details

    if is_T3_4 and grade_group in (1, 2, 3, 4):
        details.append("T3–T4, N0/NX, M0, Grade Group 1–4 → Stage IIIB.")
        return "Stage IIIB", details

    if (is_T1 or is_T2) and grade_group in (1, 2, 3, 4) and psa >= 20:
        details.append("T1–T2, N0/NX, M0, Grade Group 1–4, PSA ≥20 → Stage IIIA.")
        return "Stage IIIA", details

    if (is_T1 or is_T2) and grade_group in (3, 4) and psa < 20:
        details.append("T1–T2, N0/NX, M0, Grade Group 3–4, PSA <20 → Stage IIC.")
        return "Stage IIC", details

    if (is_T1 or is_T2) and grade_group == 2 and psa < 20:
        details.append("T1–T2, N0/NX, M0, Grade Group 2, PSA <20 → Stage IIB.")
        return "Stage IIB", details

    if grade_group == 1 and psa < 20:
        if ((is_T1 or is_T2a_) and 10 <= psa < 20) or (is_T2b_c and psa < 20):
            details.append("Grade Group 1, N0/NX, M0, PSA <20 with appropriate T → Stage IIA.")
            return "Stage IIA", details

    if grade_group == 1 and (is_T1 or is_T2a_) and psa < 10:
        details.append("cT1–cT2a, N0/NX, M0, Grade Group 1, PSA <10 → Stage I.")
        return "Stage I", details

    details.append("Does not fit standard AJCC 8th prognostic groups with the provided data.")
    return "Stage group cannot be determined", details


def get_ajcc_stage_explanation(ajcc_stage: str):
    parts = ajcc_stage.split()
    stage_code = parts[1] if len(parts) > 1 else ""
    short = None
    bullets: list[str] = []

    if stage_code == "I":
        short = "Very favorable, low-volume localized disease."
        bullets = [
            "Tumor is organ-confined with low PSA and Grade Group 1.",
            "Population-level survival for localized prostate cancer is excellent.",
        ]
    elif stage_code.startswith("II"):
        short = "Localized disease with increasing biologic risk."
        bullets = [
            "Cancer is still confined to the prostate but has higher PSA and/or Grade Group than Stage I.",
            "Population-level outcomes are generally excellent for localized disease.",
        ]
    elif stage_code.startswith("III"):
        short = "Locally advanced or biologically high-risk localized disease."
        bullets = [
            "Includes extraprostatic extension (T3–T4) and/or very high PSA or Grade Group 5 even if organ-confined.",
        ]
    elif stage_code.startswith("IVA"):
        short = "Node-positive, non-metastatic prostate cancer."
        bullets = ["Cancer has spread to regional lymph nodes but not to distant organs."]
    elif stage_code.startswith("IVB"):
        short = "Metastatic prostate cancer (distant disease)."
        bullets = ["Cancer has spread beyond the pelvis (e.g., bone, distant nodes, visceral)."]

    if short is not None:
        bullets += [
            "These are population-level generalities, not individualized predictions.",
            "For patient-specific estimates, use validated nomograms with clinical judgment.",
        ]
    return short, bullets


def _risk_key_prefix() -> str:
    return "risk__"


def rk(name: str) -> str:
    return _risk_key_prefix() + name

def _apply_manual_grade_group_from_gleason() -> None:
    """Apply helper-selected Gleason patterns to the Overall Grade Group selector."""
    try:
        ptn = int(st.session_state.get(rk("gg_help_primary"), 3))
        stn = int(st.session_state.get(rk("gg_help_secondary"), 3))
        gg, _, _ = gleason_to_grade_group(ptn, stn)
        if gg is not None:
            st.session_state[rk("manual_gg")] = int(gg)
    except Exception:
        # Best-effort helper; do not crash the app if unexpected values exist
        return



def render_biopsy_inputs():
    st.subheader("Systematic 12-Core Biopsy")
    st.caption("Detailed core entry allows for automatic calculation of positive cores percentage.")

    for code, label in CORE_SITES:
        with st.expander(f"{code}: {label}", expanded=False):
            type_key = rk(f"{code}_type")
            p_key = rk(f"{code}_p")
            s_key = rk(f"{code}_s")
            pct_key = rk(f"{code}_pct")
            epe_key = rk(f"{code}_epe")
            pni_key = rk(f"{code}_pni")

            biopsy_type = st.selectbox("Pathology result", ["Benign", "Cancer", "ASAP"], key=type_key, index=0)

            if biopsy_type == "Cancer":
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.selectbox("Primary Gleason", [3, 4, 5], index=[3, 4, 5].index(DEFAULT_PRIMARY_PATTERN), key=p_key)
                with col_b:
                    st.selectbox("Secondary Gleason", [3, 4, 5], index=[3, 4, 5].index(DEFAULT_SECONDARY_PATTERN), key=s_key)
                with col_c:
                    st.slider("% Involvement", 1, 100, DEFAULT_CORE_PERCENT, key=pct_key)
                st.checkbox("Extraprostatic extension (EPE)", key=epe_key, value=False)
                st.checkbox("Perineural invasion (PNI)", key=pni_key, value=False)
            elif biopsy_type == "ASAP":
                st.info("ASAP = atypical small acinar proliferation.")
            else:
                st.caption("No cancer identified.")

    st.subheader("Targeted Biopsy Cores (Optional)")
    
    for code, label in TARGETED_SITES:
        with st.expander(f"{code}: {label}", expanded=False):
            type_key = rk(f"{code}_type")
            p_key = rk(f"{code}_p")
            s_key = rk(f"{code}_s")
            pct_key = rk(f"{code}_pct")
            epe_key = rk(f"{code}_epe")
            pni_key = rk(f"{code}_pni")
            desc_key = rk(f"{code}_desc")

            biopsy_type = st.selectbox("Pathology result", ["Not taken", "Benign", "Cancer", "ASAP"], key=type_key, index=0)
            
            if biopsy_type == "Cancer":
                st.text_input("Site description", key=desc_key, placeholder="e.g. PIRADS 4 lesion")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.selectbox("Primary Gleason", [3, 4, 5], index=[3, 4, 5].index(DEFAULT_PRIMARY_PATTERN), key=p_key)
                with col_b:
                    st.selectbox("Secondary Gleason", [3, 4, 5], index=[3, 4, 5].index(DEFAULT_SECONDARY_PATTERN), key=s_key)
                with col_c:
                    st.slider("% Involvement", 1, 100, DEFAULT_CORE_PERCENT, key=pct_key)
                st.checkbox("Extraprostatic extension (EPE)", key=epe_key, value=False)
                st.checkbox("Perineural invasion (PNI)", key=pni_key, value=False)


def clear_risk_keys():
    kill = [k for k in st.session_state.keys() if k.startswith(_risk_key_prefix())]
    for k in kill:
        del st.session_state[k]


# ============================================================
# 2) LIFE EXPECTANCY ENGINE
# ============================================================
SSA_SOURCE_URL = "https://www.ssa.gov/oact/STATS/table4c6.html"
MSKCC_LE_URL = "https://www.mskcc.org/nomograms/prostate/male_life_expectancy"
UCSF_LEE_URL = "https://eprognosis.ucsf.edu/leeschonberg.php"
SSA_CSV_FILENAME = "table4c6.csv"
UI_MIN_AGE = 0
UI_MAX_AGE = 119


def _decode_bytes_safely(b: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("latin1", errors="ignore")


def parse_ssa_period_life_table_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("SSA life table file is missing from the app folder (table4c6.csv).")

    raw_text = _decode_bytes_safely(path.read_bytes())
    lines = raw_text.splitlines()

    start = None
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("exact,male"):
            start = i
            break
    if start is None:
        raise ValueError("Could not locate SSA table section (expected 'Exact,Male...' header).")

    reader = csv.reader(lines[start:], delimiter=",", quotechar='"')
    rows = list(reader)

    data_rows = []
    for r in rows[3:]:
        if not r or len(r) < 7:
            continue
        age_str = (r[0] or "").strip()
        if not re.fullmatch(r"\d{1,3}", age_str):
            continue

        def to_float(x: str):
            x = (x or "").strip().replace(",", "")
            try:
                return float(x)
            except ValueError:
                return None

        age = int(age_str)
        male_le = to_float(r[3]) if len(r) > 3 else None
        female_le = to_float(r[6]) if len(r) > 6 else None
        if male_le is None and female_le is None:
            continue
        data_rows.append((age, male_le, female_le))

    if not data_rows:
        raise ValueError("No usable SSA rows parsed from table4c6.csv.")

    return (
        pd.DataFrame(data_rows, columns=["age", "male_le", "female_le"])
        .drop_duplicates(subset=["age"])
        .sort_values("age")
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def load_ssa_table() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / SSA_CSV_FILENAME
    return parse_ssa_period_life_table_csv(csv_path)


def baseline_le_from_ssa(age: int, sex: str, df: pd.DataFrame) -> float | None:
    row = df.loc[df["age"] == age]
    if row.empty:
        return None
    if sex.lower().startswith("m"):
        return float(row.iloc[0]["male_le"])
    return float(row.iloc[0]["female_le"])


def le_bucket_from_years(years: float | None) -> str | None:
    if years is None:
        return None
    if years >= 10:
        return ">=10"
    if years >= 5:
        return "5-10"
    return "<5"


def get_effective_le_years() -> float | None:
    lr = st.session_state.get("le_result")
    if not lr:
        return None
    eff = lr.get("effective_years")
    return None if eff is None else float(eff)


# ============================================================
# 3) WIZARD ENGINE
# ============================================================
def badge_row(labels: List[str]) -> str:
    return "".join([f'<span class="badge">{b}</span>' for b in labels])


def render_card(title: str, subtitle: str = "", badges: Optional[List[str]] = None, highlight: bool = False):
    b = badge_row(badges or [])
    hi_class = "hi" if highlight else ""
    st.markdown(
        f"""
        <div class="card {hi_class}">
          <div class="title">{title}</div>
          <div class="subtitle">{subtitle}</div>
          <div>{b}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def goto_section(name: str):
    # Don't write to a widget-bound key after instantiation; use a pending key instead.
    st.session_state["_pending_eval_section"] = name
    st.rerun()


def reset_all_wizards():
    keys = [k for k in st.session_state.keys() if k.startswith("wiz__") or k.startswith("wizchoice__")]
    for k in keys:
        del st.session_state[k]


@dataclass
class Choice:
    label: str
    next_id: str


@dataclass
class Node:
    id: str
    kind: str  # "question" or "result" or "info"
    title: str
    prompt: str = ""
    choices: Optional[List[Choice]] = None
    answer_key: Optional[str] = None
    body_md: str = ""
    callout: str = ""  # "", "info", "success", "warning", "error"
    nav_to: Optional[List[Tuple[str, str]]] = None
    continue_to: Optional[str] = None

    # AUTO-ROUTING (keeps diagram shape; wizard UI just indicates chosen branch)
    auto_mode: Optional[str] = None  # "le_threshold"
    auto_threshold: Optional[float] = None  # e.g., 10 or 5
    auto_symptom_key: Optional[str] = None  # optional checkbox key to treat as "high branch"


@dataclass
class WizardSpec:
    id: str
    title: str
    subtitle: str
    start_id: str
    nodes: Dict[str, Node]


def wizard_state_key(wiz_id: str) -> str:
    return f"wiz__{wiz_id}"


def wizard_init(spec: WizardSpec):
    k = wizard_state_key(spec.id)
    if k not in st.session_state:
        st.session_state[k] = {"current": spec.start_id, "history": [], "answers": {}}


def wizard_reset(spec: WizardSpec):
    k = wizard_state_key(spec.id)
    st.session_state[k] = {"current": spec.start_id, "history": [], "answers": {}}


def wizard_back(spec: WizardSpec):
    k = wizard_state_key(spec.id)
    state = st.session_state[k]
    if state["history"]:
        state["current"] = state["history"].pop()


def wizard_set_answer(spec: WizardSpec, key: str, value: str):
    k = wizard_state_key(spec.id)
    st.session_state[k]["answers"][key] = value


def wizard_get_answers(spec: WizardSpec) -> Dict[str, str]:
    k = wizard_state_key(spec.id)
    return st.session_state[k]["answers"]


# ============================================================
# GRAPHVIZ: render a WizardSpec as a decision-tree diagram
# ============================================================
def _dot_escape(s: str) -> str:
    # Basic HTML escaping for labels
    return html.escape(s, quote=True)


def _wrap_label(s: str, width: int = 18) -> str:
    """Wraps text with <br/> tags for HTML-like labels."""
    s = (s or "").strip()
    if len(s) <= width:
        return _dot_escape(s)
    
    words = s.split()
    lines, cur = [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(_dot_escape(cur))
            cur = w
    if cur:
        lines.append(_dot_escape(cur))
    return "<br/>".join(lines)


def _dot_id(raw: str) -> str:
    raw = (raw or "").strip()
    raw = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    if not raw:
        raw = "node"
    if raw[0].isdigit():
        raw = "n_" + raw
    return raw


def wizard_to_dot(
    spec: WizardSpec,
    state: dict | None = None,
    rankdir: str = "TB",
    highlight_edge: tuple[str, str] | None = None,  # (src_node_id, dst_node_id) in spec node_id space
) -> str:
    visited = set()
    current = None
    if state:
        current = state.get("current")
        visited = set(state.get("history", [])) | ({current} if current else set())

    lines: list[str] = []
    lines.append("digraph G {")
    lines.append(f"  rankdir={rankdir};")
    # Clean orthogonal layout
    lines.append('  graph [fontsize=10, splines="ortho", nodesep=0.6, ranksep=0.6, fontname="Arial"];')
    lines.append('  node  [fontsize=10, shape=box, style="filled", fillcolor="#F8F9FA", fontname="Arial"];')
    lines.append('  edge  [fontsize=9, fontname="Arial"];')

    for node_id, node in spec.nodes.items():
        nid = _dot_id(f"{spec.id}__{node_id}")

        if node.kind == "question":
            shape = "diamond"
            # Slightly different color for decision nodes
            fill = "#FFD700" if (current and node_id == current) else "#E0E0E0"
        elif node.kind == "info":
            shape = "oval"
            fill = "#FFD700" if (current and node_id == current) else "#F8F9FA"
        else:
            shape = "box"
            fill = "#FFD700" if (current and node_id == current) else "#F8F9FA"

        if node_id in visited and node_id != current:
             fill = "#D3D3D3" # Visited path is grey

        # Use HTML-like label (<...>) to interpret <br/> correctly
        label_html = _wrap_label(node.title, 22)
        
        attrs = [f'shape={shape}', f'label=<{label_html}>', f'fillcolor="{fill}"']
        lines.append(f"  {nid} [{', '.join(attrs)}];")

    for node_id, node in spec.nodes.items():
        src = _dot_id(f"{spec.id}__{node_id}")

        if node.kind == "question" and node.choices:
            for ch in node.choices:
                dst = _dot_id(f"{spec.id}__{ch.next_id}")
                elab_html = _wrap_label(ch.label, 20)
                edge_attrs = [f'label=<{elab_html}>']
                
                if highlight_edge and highlight_edge == (node_id, ch.next_id):
                    edge_attrs += ['color="#B22222"', "penwidth=2.5"]
                
                lines.append(f"  {src} -> {dst} [{', '.join(edge_attrs)}];")

        if node.continue_to:
            dst = _dot_id(f"{spec.id}__{node.continue_to}")
            edge_attrs = ['label="Continue"']
            if highlight_edge and highlight_edge == (node_id, node.continue_to):
                edge_attrs += ['color="#B22222"', "penwidth=2.5"]
            lines.append(f"  {src} -> {dst} [{', '.join(edge_attrs)}];")

    lines.append("}")
    return "\n".join(lines)


def _auto_route_for_node(node: Node) -> tuple[bool, str, str, tuple[str, str] | None]:
    """
    Returns:
      (handled, selected_label, next_id, highlight_edge)
    If handled=False, caller should show normal radio.
    """
    if node.kind != "question" or not node.choices or not node.auto_mode:
        return False, "", "", None

    eff = get_effective_le_years()
    if eff is None:
        return False, "", "", None

    threshold = float(node.auto_threshold or 0.0)
    symptomatic = False
    if node.auto_symptom_key:
        symptomatic = bool(st.session_state.get(node.auto_symptom_key, False))

    # Convention: first choice = higher survival / more aggressive path; second = lower survival path
    choose_high = (eff >= threshold) or symptomatic
    chosen = node.choices[0] if choose_high else node.choices[1]
    return True, chosen.label, chosen.next_id, (node.id, chosen.next_id)


def run_wizard(spec: WizardSpec, badges: Optional[List[str]] = None):
    wizard_init(spec)
    k = wizard_state_key(spec.id)
    state = st.session_state[k]
    node = spec.nodes[state["current"]]

    render_card(spec.title, spec.subtitle, badges=badges or [], highlight=False)

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 2.8, 2.0])
    with c1:
        st.markdown(f"<span class='step'>Step {len(state['history'])+1}</span>", unsafe_allow_html=True)
    with c2:
        if st.button("⬅ Back", disabled=(len(state["history"]) == 0), key=f"{spec.id}__back"):
            wizard_back(spec)
            st.rerun()
    with c3:
        ans = wizard_get_answers(spec)
        if ans:
            summary = " · ".join([f"{k}: {v}" for k, v in ans.items()])
            st.markdown(f"<span class='muted tiny'>Your selections: {summary}</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='muted tiny'>Make selections step-by-step.</span>", unsafe_allow_html=True)
    with c4:
        if st.button("↺ Restart", key=f"{spec.id}__restart"):
            wizard_reset(spec)
            st.rerun()

    st.divider()

    if node.kind == "question":
        st.markdown(f"### {node.title}")
        if node.prompt:
            st.markdown(node.prompt)

        handled, sel_label, next_id, _ = _auto_route_for_node(node)
        if handled:
            eff = get_effective_le_years()
            if eff is not None:
                st.info(f"Based on stored life expectancy **{eff:.1f} years**, the pathway follows: **{sel_label}**.")
            else:
                st.info(f"Auto-selected branch: **{sel_label}**.")

            if node.answer_key:
                wizard_set_answer(spec, node.answer_key, sel_label)

            if st.button("Continue →", key=f"{spec.id}__{node.id}__auto_continue"):
                state["history"].append(node.id)
                state["current"] = next_id
                st.rerun()
            return

        # Manual choice (only if LE not stored or auto not configured)
        choices = node.choices or []
        labels = [c.label for c in choices]
        choice_key = f"wizchoice__{spec.id}__{node.id}"
        selected = st.radio("Choose one:", labels, index=0, horizontal=False, key=choice_key)
        next_id = next(c.next_id for c in choices if c.label == selected)

        if node.answer_key:
            wizard_set_answer(spec, node.answer_key, selected)

        if st.button("Next →", key=f"{spec.id}__{node.id}__next"):
            state["history"].append(node.id)
            state["current"] = next_id
            st.rerun()

    else:
        st.markdown(f"### {node.title}")

        if node.callout == "info":
            st.info(node.body_md)
        elif node.callout == "success":
            st.success(node.body_md)
        elif node.callout == "warning":
            st.warning(node.body_md)
        elif node.callout == "error":
            st.error(node.body_md)
        else:
            st.markdown(node.body_md)

        if node.nav_to:
            st.markdown("#### Quick links")
            btn_cols = st.columns(min(3, len(node.nav_to)))
            for i, (lbl, target) in enumerate(node.nav_to):
                with btn_cols[i % len(btn_cols)]:
                    if st.button(lbl, key=f"{spec.id}__nav__{node.id}__{i}"):
                        goto_section(target)

        if node.continue_to:
            if st.button("Continue →", key=f"{spec.id}__cont__{node.id}"):
                state["history"].append(node.id)
                state["current"] = node.continue_to
                st.rerun()


# -------------------------
# Wizards (Strictly aligned to PROS-3 through PROS-15 DOT files)
# -------------------------
def wiz_low_risk(wiz_id: str) -> WizardSpec:
    nodes = {
        "le": Node(
            id="le",
            kind="question",
            title="Life expectancy",
            prompt="Choose the life expectancy track (or auto-select).",
            answer_key="Life expectancy",
            choices=[Choice("≥ 10 years", "choose_track"), Choice("< 10 years", "obs_result")],
            auto_mode="le_threshold",
            auto_threshold=10,
        ),
        "obs_result": Node(
            id="obs_result",
            kind="result",
            title="Observation",
            callout="info",
            body_md="Observation (PROS-8C).",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "choose_track": Node(
            id="choose_track",
            kind="question",
            title="Initial therapy options",
            prompt="Select initial treatment.",
            answer_key="Initial strategy",
            choices=[
                Choice("Active surveillance (preferred)", "as_prog"),
                Choice("Radiation therapy (RT)", "rt_bcr"),
                Choice("Radical prostatectomy (RP)", "rp_psa"),
            ],
        ),
        "as_prog": Node(
            id="as_prog",
            kind="question",
            title="Active surveillance",
            prompt="Monitoring. Is there progressive disease?",
            answer_key="AS status",
            choices=[Choice("No progression", "as_stable"), Choice("Progressive disease", "as_progress")],
        ),
        "as_stable": Node(
            id="as_stable",
            kind="result",
            title="Continue active surveillance",
            callout="success",
            body_md="Continue surveillance.",
            nav_to=[("Open Active surveillance program", "Active surveillance program")],
        ),
        "as_progress": Node(
            id="as_progress",
            kind="result",
            title="Initial risk stratification & staging workup",
            callout="warning",
            body_md="Proceed to PROS-2 (Initial risk stratification & staging workup).",
        ),
        "rt_bcr": Node(
            id="rt_bcr",
            kind="question",
            title="After RT",
            prompt="Is there biochemical recurrence?",
            answer_key="Post-RT status",
            choices=[Choice("No", "rt_no_bcr"), Choice("Yes", "rt_yes_bcr")],
        ),
        "rt_no_bcr": Node(
            id="rt_no_bcr",
            kind="result",
            title="Monitoring",
            callout="success",
            body_md="Proceed to PROS-8A (Monitoring).",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "rt_yes_bcr": Node(
            id="rt_yes_bcr",
            kind="result",
            title="RT recurrence",
            callout="warning",
            body_md="Proceed to PROS-10 (RT recurrence).",
        ),
        "rp_psa": Node(
            id="rp_psa",
            kind="question",
            title="After RP",
            prompt="What is the PSA status?",
            answer_key="Post-RP PSA",
            choices=[Choice("Undetectable", "rp_adverse"), Choice("PSA persistence", "rp_persist")],
        ),
        "rp_persist": Node(
            id="rp_persist",
            kind="result",
            title="RP PSA persistence/recurrence",
            callout="warning",
            body_md="Proceed to PROS-9B.",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "rp_adverse": Node(
            id="rp_adverse",
            kind="question",
            title="Adverse features / LN mets?",
            prompt="Are there adverse features or lymph node metastases?",
            answer_key="Adverse features",
            choices=[Choice("No", "rp_no_adv"), Choice("Yes", "rp_yes_adv")],
        ),
        "rp_no_adv": Node(
            id="rp_no_adv",
            kind="result",
            title="Monitoring",
            callout="success",
            body_md="Proceed to PROS-8B (Monitoring).",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "rp_yes_adv": Node(
            id="rp_yes_adv",
            kind="result",
            title="Monitoring or Treatment",
            callout="warning",
            body_md="Proceed to PROS-9 (Monitoring preferred or consider treatment).",
        ),
    }
    return WizardSpec(
        id=wiz_id,
        title="Low-risk (PROS-3)",
        subtitle="Matches NCCN PROS-3 flowchart.",
        start_id="le",
        nodes=nodes,
    )


def wiz_favorable_intermediate(wiz_id: str) -> WizardSpec:
    nodes = {
        "le": Node(
            id="le",
            kind="question",
            title="Life expectancy",
            prompt="Choose life expectancy.",
            answer_key="Life expectancy",
            choices=[Choice("> 10 years", "gt10_track"), Choice("5–10 years", "y5_10_track")],
            auto_mode="le_threshold",
            auto_threshold=10,
        ),
        "gt10_track": Node(
            id="gt10_track",
            kind="question",
            title="Initial therapy (>10y)",
            prompt="Select initial treatment.",
            answer_key="Initial strategy",
            choices=[
                Choice("Active surveillance (selected)", "as_prog"),
                Choice("Radical prostatectomy (RP)", "rp_psa"),
                Choice("Radiation therapy (RT)", "rt_bcr"),
            ],
        ),
        "y5_10_track": Node(
            id="y5_10_track",
            kind="question",
            title="Initial therapy (5–10y)",
            prompt="Select initial treatment.",
            answer_key="Initial strategy",
            choices=[
                Choice("Observation (preferred)", "obs_mon"),
                Choice("Radiation therapy (RT)", "rt2_mon"),
            ],
        ),
        "obs_mon": Node(
            id="obs_mon",
            kind="result",
            title="Monitoring",
            callout="success",
            body_md="Proceed to PROS-8C (Monitoring).",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "rt2_mon": Node(
            id="rt2_mon",
            kind="result",
            title="Monitoring",
            callout="success",
            body_md="Proceed to PROS-8D (Monitoring).",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        # Branches from GT10
        "as_prog": Node(
            id="as_prog",
            kind="question",
            title="Active surveillance",
            prompt="Monitoring. Is there progressive disease?",
            choices=[Choice("No progression", "as_stable"), Choice("Progressive disease", "as_progress")],
        ),
        "as_stable": Node(
            id="as_stable",
            kind="result",
            title="Continue active surveillance",
            callout="success",
            body_md="Continue surveillance.",
            nav_to=[("Open Active surveillance program", "Active surveillance program")],
        ),
        "as_progress": Node(
            id="as_progress",
            kind="result",
            title="Initial risk stratification & staging workup",
            callout="warning",
            body_md="Proceed to PROS-2.",
        ),
        "rt_bcr": Node(
            id="rt_bcr",
            kind="question",
            title="After RT",
            prompt="Is there biochemical recurrence?",
            choices=[Choice("No", "rt_no_bcr"), Choice("Yes", "rt_yes_bcr")],
        ),
        "rt_no_bcr": Node(
            id="rt_no_bcr",
            kind="result",
            title="Monitoring",
            callout="success",
            body_md="Proceed to PROS-8B (Monitoring).",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "rt_yes_bcr": Node(
            id="rt_yes_bcr",
            kind="result",
            title="RT recurrence",
            callout="warning",
            body_md="Proceed to PROS-10.",
        ),
        "rp_psa": Node(
            id="rp_psa",
            kind="question",
            title="After RP",
            prompt="What is the PSA status?",
            choices=[Choice("Undetectable", "rp_adverse"), Choice("PSA persistence", "rp_persist")],
        ),
        "rp_persist": Node(
            id="rp_persist",
            kind="result",
            title="RP PSA persistence/recurrence",
            callout="warning",
            body_md="Proceed to PROS-9B.",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "rp_adverse": Node(
            id="rp_adverse",
            kind="question",
            title="Adverse features / LN mets?",
            prompt="Are there adverse features or lymph node metastases?",
            choices=[Choice("No", "rp_no_adv"), Choice("Yes", "rp_yes_adv")],
        ),
        "rp_no_adv": Node(
            id="rp_no_adv",
            kind="result",
            title="Monitoring",
            callout="success",
            body_md="Proceed to PROS-8A (Monitoring).",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "rp_yes_adv": Node(
            id="rp_yes_adv",
            kind="result",
            title="Monitoring or Treatment",
            callout="warning",
            body_md="Proceed to PROS-9 (Monitoring preferred or consider treatment).",
        ),
    }
    return WizardSpec(
        id=wiz_id,
        title="Favorable Intermediate-Risk (PROS-4)",
        subtitle="Matches NCCN PROS-4 flowchart.",
        start_id="le",
        nodes=nodes,
    )


def wiz_unfavorable_intermediate(wiz_id: str) -> WizardSpec:
    nodes = {
        "le": Node(
            id="le",
            kind="question",
            title="Life expectancy",
            prompt="Choose life expectancy.",
            answer_key="Life expectancy",
            choices=[Choice("> 10 years", "gt10_track"), Choice("5–10 years", "y5_10_track")],
            auto_mode="le_threshold",
            auto_threshold=10,
        ),
        "gt10_track": Node(
            id="gt10_track",
            kind="question",
            title="Initial therapy (>10y)",
            prompt="Select initial treatment.",
            answer_key="Initial strategy",
            choices=[
                Choice("Radical prostatectomy (RP)", "rp_psa"),
                Choice("RT + ADT (4–6 mo)", "rt_bcr"),
            ],
        ),
        "y5_10_track": Node(
            id="y5_10_track",
            kind="result",
            title="Observation",
            callout="info",
            body_md="Observation -> PROS-8C (Monitoring).",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "rt_bcr": Node(
            id="rt_bcr",
            kind="question",
            title="After RT+ADT",
            prompt="Is there biochemical recurrence?",
            choices=[Choice("No", "rt_no_bcr"), Choice("Yes", "rt_yes_bcr")],
        ),
        "rt_no_bcr": Node(
            id="rt_no_bcr",
            kind="result",
            title="Monitoring",
            callout="success",
            body_md="Proceed to PROS-8B (Monitoring).",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "rt_yes_bcr": Node(
            id="rt_yes_bcr",
            kind="result",
            title="RT recurrence",
            callout="warning",
            body_md="Proceed to PROS-10.",
        ),
        "rp_psa": Node(
            id="rp_psa",
            kind="question",
            title="After RP",
            prompt="What is the PSA status?",
            choices=[Choice("Undetectable", "rp_adverse"), Choice("PSA persistence", "rp_persist")],
        ),
        "rp_persist": Node(
            id="rp_persist",
            kind="result",
            title="RP PSA persistence/recurrence",
            callout="warning",
            body_md="Proceed to PROS-9B.",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "rp_adverse": Node(
            id="rp_adverse",
            kind="question",
            title="Adverse features / LN mets?",
            prompt="Are there adverse features or lymph node metastases?",
            choices=[Choice("No", "rp_no_adv"), Choice("Yes", "rp_yes_adv")],
        ),
        "rp_no_adv": Node(
            id="rp_no_adv",
            kind="result",
            title="Monitoring",
            callout="success",
            body_md="Proceed to PROS-8A (Monitoring).",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "rp_yes_adv": Node(
            id="rp_yes_adv",
            kind="result",
            title="Monitoring or Treatment",
            callout="warning",
            body_md="Proceed to PROS-9 (Monitoring preferred or consider treatment).",
        ),
    }
    return WizardSpec(
        id=wiz_id,
        title="Unfavorable Intermediate-Risk (PROS-5)",
        subtitle="Matches NCCN PROS-5 flowchart.",
        start_id="le",
        nodes=nodes,
    )


def wiz_high_very_high(wiz_id: str) -> WizardSpec:
    nodes = {
        "bucket": Node(
            id="bucket",
            kind="question",
            title="Expected survival",
            prompt="Select context.",
            answer_key="Context",
            choices=[
                Choice("> 5y OR symptomatic", "high_track"),
                Choice("≤ 5y AND asymptomatic", "low_track"),
            ],
            auto_mode="le_threshold",
            auto_threshold=5,
            auto_symptom_key="pc_symptoms",
        ),
        "low_track": Node(
            id="low_track",
            kind="question",
            title="Treatment options (≤5y Asymp)",
            prompt="Select option.",
            choices=[Choice("Observation", "obs_symp"), Choice("RT + ADT", "obs_symp")],
        ),
        "obs_symp": Node(
            id="obs_symp",
            kind="question",
            title="Follow-up",
            prompt="Is there symptomatic progression?",
            choices=[Choice("No", "continue_mon"), Choice("Yes (Symptomatic progression)", "prog_met")],
        ),
        "continue_mon": Node(
            id="continue_mon",
            kind="result",
            title="Continue",
            callout="success",
            body_md="Continue observation or management.",
        ),
        "prog_met": Node(
            id="prog_met",
            kind="result",
            title="Metastatic progression",
            callout="warning",
            body_md="Proceed to PROS-13 (Metachronous oligo), PROS-14 (Low-vol), or PROS-15 (High-vol).",
        ),
        "high_track": Node(
            id="high_track",
            kind="question",
            title="Initial therapy (>5y/Symp)",
            prompt="Select initial treatment.",
            choices=[
                Choice("RT + ADT (12–36 mo) ± abiraterone", "rt_bcr"),
                Choice("RP (select patients)", "rp_psa"),
            ],
        ),
        "rt_bcr": Node(
            id="rt_bcr",
            kind="question",
            title="After RT+ADT",
            prompt="Is there biochemical recurrence?",
            choices=[Choice("No", "rt_no_bcr"), Choice("Yes", "rt_yes_bcr")],
        ),
        "rt_no_bcr": Node(
            id="rt_no_bcr",
            kind="result",
            title="Monitoring",
            callout="success",
            body_md="Proceed to PROS-8A (Monitoring).",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "rt_yes_bcr": Node(
            id="rt_yes_bcr",
            kind="result",
            title="RT recurrence",
            callout="warning",
            body_md="Proceed to PROS-10.",
        ),
        "rp_psa": Node(
            id="rp_psa",
            kind="question",
            title="After RP",
            prompt="What is the PSA status?",
            choices=[Choice("Undetectable", "rp_adverse"), Choice("PSA persistence", "rp_persist")],
        ),
        "rp_persist": Node(
            id="rp_persist",
            kind="result",
            title="RP PSA persistence/recurrence",
            callout="warning",
            body_md="Proceed to PROS-9B.",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "rp_adverse": Node(
            id="rp_adverse",
            kind="question",
            title="Adverse features / LN mets?",
            prompt="Are there adverse features or lymph node metastases?",
            choices=[Choice("No", "rp_no_adv"), Choice("Yes", "rp_yes_adv")],
        ),
        "rp_no_adv": Node(
            id="rp_no_adv",
            kind="result",
            title="Monitoring",
            callout="success",
            body_md="Proceed to PROS-8B (Monitoring).",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "rp_yes_adv": Node(
            id="rp_yes_adv",
            kind="result",
            title="Monitoring or Treatment",
            callout="warning",
            body_md="Proceed to PROS-9 (Monitoring preferred or consider treatment).",
        ),
    }
    return WizardSpec(
        id=wiz_id,
        title="High / Very-High Risk (PROS-6)",
        subtitle="Matches NCCN PROS-6 flowchart.",
        start_id="bucket",
        nodes=nodes,
    )


def wiz_regional_n1m0(wiz_id: str) -> WizardSpec:
    nodes = {
        "bucket": Node(
            id="bucket",
            kind="question",
            title="Expected survival",
            prompt="Select context.",
            answer_key="Context",
            choices=[
                Choice("> 5y OR symptomatic", "high_track"),
                Choice("≤ 5y AND asymptomatic", "low_track"),
            ],
            auto_mode="le_threshold",
            auto_threshold=5,
            auto_symptom_key="pc_symptoms",
        ),
        "low_track": Node(
            id="low_track",
            kind="question",
            title="Treatment options (≤5y Asymp)",
            prompt="Select option.",
            choices=[Choice("Observation", "obs_symp"), Choice("RT", "obs_symp"), Choice("ADT ± RT", "obs_symp")],
        ),
        "obs_symp": Node(
            id="obs_symp",
            kind="question",
            title="Follow-up",
            prompt="Is there symptomatic progression?",
            choices=[Choice("No", "continue_mon"), Choice("Yes (Symptomatic)", "prog_met")],
        ),
        "continue_mon": Node(
            id="continue_mon",
            kind="result",
            title="Continue",
            callout="success",
            body_md="Continue observation or management.",
        ),
        "prog_met": Node(
            id="prog_met",
            kind="result",
            title="Metastatic progression",
            callout="warning",
            body_md="Proceed to PROS-13/14/15 or PROS-17 (M1 CRPC).",
        ),
        "high_track": Node(
            id="high_track",
            kind="question",
            title="Initial therapy (>5y/Symp)",
            prompt="Select initial treatment.",
            choices=[
                Choice("RT + ADT + abiraterone (preferred)", "rt_bcr"),
                Choice("ADT + abiraterone", "sys_mon"),
                Choice("RP (select patients)", "rp_psa"),
            ],
        ),
        "rt_bcr": Node(
            id="rt_bcr",
            kind="question",
            title="After RT+ADT",
            prompt="Is there biochemical recurrence?",
            choices=[Choice("No", "rt_no_bcr"), Choice("Yes", "rt_yes_bcr")],
        ),
        "rt_no_bcr": Node(
            id="rt_no_bcr",
            kind="result",
            title="Monitoring",
            callout="success",
            body_md="Proceed to PROS-8A (Monitoring).",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "rt_yes_bcr": Node(
            id="rt_yes_bcr",
            kind="result",
            title="RT recurrence",
            callout="warning",
            body_md="Proceed to PROS-10.",
        ),
        "sys_mon": Node(
            id="sys_mon",
            kind="result",
            title="Monitoring",
            callout="success",
            body_md="Proceed to PROS-8B (Monitoring).",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "rp_psa": Node(
            id="rp_psa",
            kind="question",
            title="After RP",
            prompt="What is the PSA status?",
            choices=[Choice("Undetectable", "rp_adverse"), Choice("PSA persistence", "rp_persist")],
        ),
        "rp_persist": Node(
            id="rp_persist",
            kind="result",
            title="RP PSA persistence/recurrence",
            callout="warning",
            body_md="Proceed to PROS-9B.",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "rp_adverse": Node(
            id="rp_adverse",
            kind="question",
            title="Adverse features / LN mets?",
            prompt="Are there adverse features or lymph node metastases?",
            choices=[Choice("No", "rp_no_adv"), Choice("Yes", "rp_yes_adv")],
        ),
        "rp_no_adv": Node(
            id="rp_no_adv",
            kind="result",
            title="Monitoring",
            callout="success",
            body_md="Proceed to PROS-8C (Monitoring).",
            nav_to=[("Open Monitoring & recurrence", "Monitoring & recurrence")],
        ),
        "rp_yes_adv": Node(
            id="rp_yes_adv",
            kind="result",
            title="Monitoring or Treatment",
            callout="warning",
            body_md="Proceed to PROS-9 (Monitoring preferred or consider treatment).",
        ),
    }
    return WizardSpec(
        id=wiz_id,
        title="Regional (N1, M0) (PROS-7)",
        subtitle="Matches NCCN PROS-7 flowchart.",
        start_id="bucket",
        nodes=nodes,
    )


def wiz_monitoring_recurrence(wiz_id: str) -> WizardSpec:
    nodes = {
        "ctx": Node(
            id="ctx",
            kind="question",
            title="Scenario",
            prompt="Select a scenario (PROS-8).",
            choices=[
                Choice("Post-RP", "post_rp"),
                Choice("Post-RT", "post_rt"),
                Choice("Radiographic mets w/o PSA recurrence", "radio"),
                Choice("N1 on ADT or Localized on Obs", "n1_loc"),
            ],
        ),
        "post_rp": Node(
            id="post_rp",
            kind="question",
            title="Post-RP",
            prompt="PSA persistence/recurrence?",
            choices=[Choice("Yes", "pros9")],
        ),
        "post_rt": Node(
            id="post_rt",
            kind="question",
            title="Post-RT",
            prompt="PSA recurrence or +DRE?",
            choices=[Choice("Yes", "pros10")],
        ),
        "radio": Node(
            id="radio",
            kind="question",
            title="Radiographic",
            prompt="Biopsy mets site?",
            choices=[Choice("Yes", "biopsy")],
        ),
        "biopsy": Node(
            id="biopsy",
            kind="result",
            title="Mets confirmed",
            callout="warning",
            body_md="Proceed to PROS-13 / PROS-14 / PROS-15.",
        ),
        "n1_loc": Node(
            id="n1_loc",
            kind="question",
            title="Follow-up",
            prompt="Progression?",
            choices=[Choice("M0", "pros16"), Choice("M1", "pros17")],
        ),
        "pros9": Node(id="pros9", kind="result", title="PROS-9", callout="warning", body_md="Proceed to PROS-9."),
        "pros10": Node(id="pros10", kind="result", title="PROS-10", callout="warning", body_md="Proceed to PROS-10."),
        "pros16": Node(id="pros16", kind="result", title="PROS-16", callout="warning", body_md="Proceed to PROS-16 (M0 CRPC)."),
        "pros17": Node(id="pros17", kind="result", title="PROS-17", callout="warning", body_md="Proceed to PROS-17 (M1 CRPC)."),
    }
    return WizardSpec(
        id=wiz_id,
        title="Monitoring & Recurrence (PROS-8)",
        subtitle="Matches NCCN PROS-8 flowchart.",
        start_id="ctx",
        nodes=nodes,
    )


def wiz_metastatic_cspc(wiz_id: str) -> WizardSpec:
    nodes = {
        "pattern": Node(
            id="pattern",
            kind="question",
            title="Disease volume",
            prompt="Select volume/pattern.",
            choices=[Choice("Low-volume / Oligometastatic (PROS-14)", "low_vol"), Choice("High-volume (PROS-15)", "high_vol")],
        ),
        "low_vol": Node(
            id="low_vol",
            kind="question",
            title="Presentation",
            prompt="Synchronous or Metachronous?",
            choices=[Choice("Synchronous", "synch"), Choice("Metachronous", "meta")],
        ),
        "synch": Node(
            id="synch",
            kind="result",
            title="Treatment (PROS-14)",
            callout="success",
            body_md="**ADT + one of:**\n- Abiraterone / Apalutamide / Enzalutamide\n- Docetaxel (+/- abiraterone/darolutamide)\n- EBRT to primary tumor",
        ),
        "meta": Node(
            id="meta",
            kind="result",
            title="Treatment (PROS-14)",
            callout="success",
            body_md="**ADT + one of:**\n- Abiraterone / Apalutamide / Enzalutamide\n- Darolutamide",
        ),
        "high_vol": Node(
            id="high_vol",
            kind="result",
            title="Treatment (PROS-15)",
            callout="success",
            body_md="**ADT + one of:**\n- Docetaxel + (abiraterone or darolutamide)\n- Abiraterone / Apalutamide / Enzalutamide",
        ),
    }
    return WizardSpec(
        id=wiz_id,
        title="Metastatic CSPC (PROS-14/15)",
        subtitle="Matches NCCN PROS-14/15.",
        start_id="pattern",
        nodes=nodes,
    )


def risk_to_wizard_key(risk_group: str | None, subcategory: str | None, disease_category: str | None) -> str:
    if disease_category:
        if disease_category.startswith("Node-positive"):
            return "regional_n1m0"
        if disease_category.startswith("Metastatic"):
            return "high"

    if not risk_group:
        return "low"

    rg = risk_group.strip()
    sub = (subcategory or "").strip()

    if rg == "Low":
        return "low"
    if rg == "Intermediate" and sub == "Favorable":
        return "int_fav"
    if rg == "Intermediate" and sub == "Unfavorable":
        return "int_unfav"
    if rg == "High":
        return "high"
    if rg == "Very high":
        return "very_high"
    if "Metastatic/regional" in rg:
        return "regional_n1m0"

    return "low"


# ============================================================


# ============================================================
# GLUE / INTEGRATION HELPERS
# ============================================================
def _current_psa_from_screening() -> float | None:
    """Return the *current* PSA from the screening wizard, if available.

    Mirrors screening logic: repeated PSA overrides initial PSA, and only
    returns a value if PSA was marked as performed.
    """
    if not st.session_state.get('psa_done', False):
        return None
    val = st.session_state.get('psa_repeat_value') if st.session_state.get('psa_repeated') else st.session_state.get('psa_value')
    try:
        return None if val is None else float(val)
    except Exception:
        return None


def _apply_handoff_psa_to_post_biopsy() -> None:
    """Apply Screening→Post-biopsy PSA transfer in a Streamlit-safe way.

    - Does NOT write to the PSA widget key directly (avoids Streamlit yellow warning).
    - Clears the PSA widget key *before* it is created so the transferred PSA becomes the default.
    - Runs only when _handoff_psa_apply is True (set at the exact moment of handoff).
    """
    if not st.session_state.get("_handoff_psa_apply", False):
        return

    psa = st.session_state.get("_handoff_psa", None)
    if psa is None:
        psa = _current_psa_from_screening()

    if psa is None:
        st.session_state["_handoff_psa_apply"] = False
        return

    try:
        psa = float(psa)
    except Exception:
        st.session_state["_handoff_psa_apply"] = False
        return

    # Clear the post-biopsy PSA widget key so the next render uses our default.
    try:
        st.session_state.pop("risk__psa", None)
    except Exception:
        pass

    st.session_state["_pb_prefill_psa_value"] = psa
    st.session_state["_handoff_psa_apply"] = False


def prefill_post_biopsy_from_screening(force: bool = False) -> None:
    """Best-effort prefill when entering post-biopsy module.

    We keep this limited to demographics (age/sex). PSA transfer is handled
    explicitly on entry from the screening wizard to avoid Streamlit key conflicts.
    """
    # Age → life expectancy page
    if 'le_age' not in st.session_state and 'age' in st.session_state:
        try:
            st.session_state['le_age'] = int(st.session_state['age'])
        except Exception:
            pass

    # Sex: prostate is male; keep editable in the post-biopsy module if desired
    if 'le_sex' not in st.session_state:
        st.session_state['le_sex'] = 'Male'


def render_sidebar_embedded(render_sidebar_fn) -> None:
    """Adds a small 'Back to screening' control above the post-biopsy sidebar."""
    with st.sidebar:
        st.markdown('### 🔙 Return')
        if st.button('⬅ Back to screening wizard', use_container_width=True):
            # Return to biopsy-result step
            st.session_state['wizard_step'] = 3
            # Reset post-biopsy workflow to step 1 for next entry
            st.session_state['workflow_step'] = 1
            st.rerun()
        st.divider()
    # Render the original post-biopsy sidebar beneath the return control.
    render_sidebar_fn()



def post_biopsy_main() -> None:
    # SIDEBAR & HEADER
    # ============================================================
    _ensure_workflow_state()

    # Visual Progress Indicator
    step = int(st.session_state["workflow_step"])
    steps_labels = ["Risk", "Life Expectancy", "Pathways", "Export"]

    st.markdown(f"## {PB_APP_NAME}")
    progress_val = (step - 1) / 3.0
    st.progress(progress_val)
    st.caption(f"Step {step} of 4: {steps_labels[step-1]}")


    st.markdown('**✅ Post-biopsy mode (diagnosis confirmed)**')
    st.caption('Use this module to: (1) assign clinical/pathologic stage + risk group, (2) estimate life expectancy, (3) review pathways, (4) export a report.')
    def render_sidebar():
        with st.sidebar:
            st.header("Navigation & Tools")


            st.caption('Mode: Post-biopsy module')
            st.markdown(f"**Current Step:** {step} ({steps_labels[step-1]})")

            with st.expander("📚 Quick Reference", expanded=False):
                st.markdown("### Abbreviations")
                st.markdown("""
                - **ADT:** Androgen Deprivation Therapy
                - **AS:** Active Surveillance
                - **BCR:** Biochemical Recurrence
                - **DRE:** Digital Rectal Exam
                - **EBRT:** External Beam Radiation Therapy
                - **EPE:** Extraprostatic Extension
                - **GG:** Grade Group
                - **LND:** Lymph Node Dissection
                - **PNI:** Perineural Invasion
                - **PSA:** Prostate Specific Antigen
                - **RP:** Radical Prostatectomy
                - **RT:** Radiation Therapy
                """)

                st.divider()

                st.markdown("### Risk Groups (Simplified)")
                st.markdown("""
                - **Low:** T1-T2a, GG1, PSA <10
                - **Int (Fav):** 1 IRF, GG 1-2, <50% cores
                - **Int (Unfav):** 2+ IRF or GG3 or >50% cores
                - **High:** T3a or GG 4-5 or PSA >20
                """)

            st.divider()

            with st.expander("🧾 Pre-biopsy workup (from screening)", expanded=False):
                hist = st.session_state.get("psa_history", [])
                if isinstance(hist, list) and hist:
                    # Show most recent 5 entries
                    try:
                        hist_sorted = sorted(hist, key=lambda x: x.get("date", ""))
                    except Exception:
                        hist_sorted = hist
                    for e in hist_sorted[-5:]:
                        d = e.get("date", "")
                        lbl = e.get("label", "PSA")
                        v = e.get("psa", "")
                        try:
                            v_str = f"{float(v):.2f}"
                        except Exception:
                            v_str = str(v)
                        st.write(f"{d} — {v_str} ng/mL ({lbl})")
                else:
                    st.caption("No PSA history saved.")

                psad = current_psad()
                pctfree = current_free_psa_pct()
                mri_done = bool(st.session_state.get("mri_done", False))
                pirads = st.session_state.get("pirads", "")

                # Compact, non-truncating display (avoid st.metric ellipsis in narrow sidebar)
                psad_str = f"{psad:.2f} ng/mL/cc" if psad is not None else "—"
                pctfree_str = f"{pctfree:.1f}%" if pctfree is not None else "—"
                mri_str = pirads if mri_done and str(pirads).strip() else ("Not done" if not mri_done else "—")

                st.markdown(f"**PSA density (PSAD):** {psad_str}")
                st.markdown(f"**% Free PSA:** {pctfree_str}")
                st.markdown(f"**MRI:** {mri_str}")

                bm = st.session_state.get("biomarker_entries", [])
                if isinstance(bm, list) and bm:
                    st.markdown("**Biomarkers**")
                    for e in bm:
                        st.write(f"{e.get('date','')} — {e.get('name','')}: {e.get('result','')} ({e.get('interpretation','')})")

            st.divider()
            st.markdown("### Session Controls")
            if st.button("Reset Application", type="secondary"):
                clear_risk_keys()
                for k in ["risk_result", "le_result", "combined_report_text", "last_pathway_dot", "last_pathway_title"]:
                    if k in st.session_state:
                        del st.session_state[k]
                reset_all_wizards()
                st.session_state["workflow_step"] = 1
                st.rerun()

            st.divider()
            st.caption(f"Author: {PB_APP_AUTHOR}")
            st.caption(f"Version: {PB_APP_VERSION}")

    render_sidebar_embedded(render_sidebar)

    st.divider()


    # ============================================================
    # STEP 1: RISK STRATIFICATION PAGE
    # ============================================================
    if step == 1:
        st.subheader("Step 1: Clinical Risk Stratification")


        st.caption('Minimum required: staging context + T/N/M + PSA + Grade Group (manual or via biopsy cores). Detailed core entry is optional but preferred when available.')
        with st.expander("ℹ️ How to use this tool", expanded=False):
            st.markdown(
                """
                1. **Select Context:** Choose Clinical or Pathologic staging.
                2. **Enter Data:** Input TNM, PSA, and Biopsy details.
                3. **Classify:** The app will calculate AJCC stage and NCCN risk group.
                """
            )

        stage_context = st.radio(
            "Staging Context:",
            ["Clinical staging (cTNM)", "Pathologic staging (pTNM)"],
            horizontal=True,
            key=rk("stage_context"),
        )

        with st.container(border=True):
            st.markdown("#### Tumor Characteristics")
            if stage_context.startswith("Clinical"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    ct_stage = st.selectbox("Clinical T Stage", T_STAGE_OPTIONS, index=T_STAGE_OPTIONS.index(DEFAULT_CT_STAGE), key=rk("ct_stage"))
                    st.caption(T_STAGE_DEFINITIONS[ct_stage])
                with c2:
                    cn_stage = st.selectbox("Clinical N Stage", N_STAGE_OPTIONS, index=N_STAGE_OPTIONS.index(DEFAULT_CN_STAGE), key=rk("cn_stage"))
                with c3:
                    cm_stage = st.selectbox("M Stage", M_STAGE_OPTIONS, index=M_STAGE_OPTIONS.index(DEFAULT_CM_STAGE), key=rk("cm_stage"))
                rp_margin = DEFAULT_MARGIN
            else:
                pc1, pc2, pc3 = st.columns(3)
                with pc1:
                    pt_stage = st.selectbox("Pathologic T Stage", PT_STAGE_OPTIONS, index=PT_STAGE_OPTIONS.index(DEFAULT_PT_STAGE), key=rk("pt_stage"))
                    st.caption(PT_STAGE_DEFINITIONS[pt_stage])
                with pc2:
                    pn_stage = st.selectbox("Pathologic N Stage", PN_STAGE_OPTIONS, index=PN_STAGE_OPTIONS.index(DEFAULT_PN_STAGE), key=rk("pn_stage"))
                with pc3:
                    cm_stage = st.selectbox("M Stage", M_STAGE_OPTIONS, index=M_STAGE_OPTIONS.index(DEFAULT_CM_STAGE), key=rk("cm_stage"))
                rp_margin = st.selectbox("Surgical Margin", MARGIN_OPTIONS, index=MARGIN_OPTIONS.index(DEFAULT_MARGIN), key=rk("rp_margin"))

        with st.container(border=True):
            st.markdown("#### Biomarkers & Grade")
            c1, c2 = st.columns(2)
            with c1:
                psa_key = rk("psa")
                # Seed a default once, and avoid Streamlit warning by NOT passing value=.
                # Clean any legacy prefill helper used in older combined versions.
                if '_pb_prefill_psa_value' in st.session_state:
                    st.session_state.pop('_pb_prefill_psa_value', None)
                if psa_key not in st.session_state:
                    seed = current_psa()
                    try:
                        seed_f = float(seed) if seed is not None else float(DEFAULT_PSA)
                    except Exception:
                        seed_f = float(DEFAULT_PSA)
                    st.session_state[psa_key] = seed_f
                psa = st.number_input("PSA (ng/mL)", min_value=0.0, step=0.1, key=psa_key)
            with c2:
                manual_gg_option = st.selectbox(
                    "Overall Grade Group",
                    ["Unknown", 1, 2, 3, 4, 5],
                    index=0,
                    key=rk("manual_gg"),
                    help="Select if detailed biopsy cores are not available."
                )
                manual_grade_group = None if manual_gg_option == "Unknown" else int(manual_gg_option)

                with st.expander("Grade Group helper (from Gleason primary + secondary)", expanded=False):
                    st.caption("Use when you only know Gleason patterns (e.g., 3+4).")
                    hp = st.selectbox("Primary Gleason pattern", [3, 4, 5], key=rk("gg_help_primary"))
                    hs = st.selectbox("Secondary Gleason pattern", [3, 4, 5], key=rk("gg_help_secondary"))
                    t5 = st.checkbox("Tertiary pattern 5 present", key=rk("gg_help_t5"))
                    gg2, gs2, _ = gleason_to_grade_group(int(hp), int(hs))
                    if gg2 is None:
                        st.warning("Invalid Gleason combination.")
                    else:
                        st.markdown(f"**Gleason {hp}+{hs}={gs2} → Grade Group {gg2}**")
                        if t5:
                            st.caption("Tertiary 5 does not change Grade Group but may worsen prognosis.")
                        st.button("Use as Overall Grade Group", on_click=_apply_manual_grade_group_from_gleason, use_container_width=True)


        has_biopsy_details = st.checkbox("Enter detailed biopsy core data", value=True, key=rk("has_biopsy_details"))

        if has_biopsy_details:
            st.info("Entering detailed core data allows for automatic calculation of positive core percentage.")
            if stage_context.startswith("Clinical"):
                render_biopsy_inputs()
            else:
                with st.expander("Pre-operative biopsy (historical)", expanded=False):
                    render_biopsy_inputs()

        st.divider()

        if st.button("Calculate Risk & Stage", type="primary"):
            systematic_cancer_cores = []
            targeted_cancer_cores = []
            asap_count = 0
            epe_count_any = 0
            pni_count_any = 0

            if has_biopsy_details:
                for code, label in CORE_SITES:
                    core_type = st.session_state.get(rk(f"{code}_type"), "Benign")
                    if core_type == "Cancer":
                        p = int(st.session_state.get(rk(f"{code}_p"), DEFAULT_PRIMARY_PATTERN))
                        s = int(st.session_state.get(rk(f"{code}_s"), DEFAULT_SECONDARY_PATTERN))
                        pct = int(st.session_state.get(rk(f"{code}_pct"), DEFAULT_CORE_PERCENT))
                        epe = bool(st.session_state.get(rk(f"{code}_epe"), False))
                        pni = bool(st.session_state.get(rk(f"{code}_pni"), False))

                        gg, gleason_score, _ = gleason_to_grade_group(p, s)
                        systematic_cancer_cores.append(
                            {
                                "code": code,
                                "label": label,
                                "primary": p,
                                "secondary": s,
                                "gleason_score": gleason_score,
                                "grade_group": gg,
                                "percent": pct,
                                "systematic": True,
                                "epe": epe,
                                "pni": pni,
                            }
                        )
                        if epe:
                            epe_count_any += 1
                        if pni:
                            pni_count_any += 1
                    elif core_type == "ASAP":
                        asap_count += 1

                for code, label in TARGETED_SITES:
                    core_type = st.session_state.get(rk(f"{code}_type"), "Not taken")
                    if core_type == "Cancer":
                        p = int(st.session_state.get(rk(f"{code}_p"), DEFAULT_PRIMARY_PATTERN))
                        s = int(st.session_state.get(rk(f"{code}_s"), DEFAULT_SECONDARY_PATTERN))
                        pct = int(st.session_state.get(rk(f"{code}_pct"), DEFAULT_CORE_PERCENT))
                        epe = bool(st.session_state.get(rk(f"{code}_epe"), False))
                        pni = bool(st.session_state.get(rk(f"{code}_pni"), False))
                        desc_txt = st.session_state.get(rk(f"{code}_desc"), "")

                        gg, gleason_score, _ = gleason_to_grade_group(p, s)
                        targeted_cancer_cores.append(
                            {
                                "code": code,
                                "label": label,
                                "primary": p,
                                "secondary": s,
                                "gleason_score": gleason_score,
                                "grade_group": gg,
                                "percent": pct,
                                "systematic": False,
                                "epe": epe,
                                "pni": pni,
                                "desc_txt": desc_txt,
                            }
                        )
                        if epe:
                            epe_count_any += 1
                        if pni:
                            pni_count_any += 1
                    elif core_type == "ASAP":
                        asap_count += 1

            all_cancer_cores = systematic_cancer_cores + targeted_cancer_cores
            positive_systematic = len(systematic_cancer_cores)
            positive_targeted = len(targeted_cancer_cores)
            percent_sys_pos = 100.0 * positive_systematic / TOTAL_SYSTEMATIC_CORES if TOTAL_SYSTEMATIC_CORES else 0.0

            if all_cancer_cores:
                grade_for_stage = max(c["grade_group"] for c in all_cancer_cores if c["grade_group"] is not None)
            else:
                grade_for_stage = manual_grade_group

            ct_stage = st.session_state.get(rk("ct_stage"), DEFAULT_CT_STAGE)
            cn_stage = st.session_state.get(rk("cn_stage"), DEFAULT_CN_STAGE)
            pt_stage = st.session_state.get(rk("pt_stage"), DEFAULT_PT_STAGE)
            pn_stage = st.session_state.get(rk("pn_stage"), DEFAULT_PN_STAGE)
            cm_stage = st.session_state.get(rk("cm_stage"), DEFAULT_CM_STAGE)

            if stage_context.startswith("Pathologic"):
                t_for_stage = pt_stage
                n_for_stage = pn_stage
            else:
                t_for_stage = ct_stage
                n_for_stage = cn_stage

            ajcc_stage, ajcc_info = classify_ajcc_stage(
                t_stage=t_for_stage, n_stage=n_for_stage, m_stage=cm_stage, psa=float(psa), grade_group=grade_for_stage
            )

            disease_category = classify_disease_category(n_stage=n_for_stage, m_stage=cm_stage)

            risk_group = None
            subcategory = None
            risk_info = []
            if stage_context.startswith("Clinical") and has_biopsy_details and all_cancer_cores and grade_for_stage is not None:
                risk_group, subcategory, risk_info = classify_nccn_risk(
                    t_stage=ct_stage,
                    n_stage=cn_stage,
                    m_stage=cm_stage,
                    grade_group=int(grade_for_stage),
                    psa=float(psa),
                    cores_positive=positive_systematic,
                    cores_total=TOTAL_SYSTEMATIC_CORES,
                )

            add_title, add_items = get_additional_evaluation_recommendations(risk_group, subcategory)
            treat_title, treat_sections = get_treatment_options(disease_category, risk_group, subcategory)

            risk_text = "NCCN risk not calculated"
            if risk_group:
                risk_text = risk_group + (f" ({subcategory} intermediate)" if subcategory else "")
            summary_line = (
                f"{t_for_stage} {n_for_stage} {cm_stage}, PSA {psa:.1f} ng/mL, "
                f"Grade Group {grade_for_stage if grade_for_stage is not None else 'unknown'}; "
                f"AJCC {ajcc_stage}, NCCN {risk_text}; "
                f"systematic cancer cores {positive_systematic}/{TOTAL_SYSTEMATIC_CORES} ({percent_sys_pos:.1f}%), "
                f"targeted cancer cores {positive_targeted}/{len(TARGETED_SITES)}."
            )

            st.session_state["risk_result"] = {
                "stage_context": stage_context,
                "t_for_stage": t_for_stage,
                "n_for_stage": n_for_stage,
                "m_for_stage": cm_stage,
                "psa": float(psa),
                "grade_group": grade_for_stage,
                "ajcc_stage": ajcc_stage,
                "ajcc_info": ajcc_info,
                "disease_category": disease_category,
                "risk_group": risk_group,
                "subcategory": subcategory,
                "risk_info": risk_info,
                "add_title": add_title,
                "add_items": add_items,
                "treat_title": treat_title,
                "treat_sections": treat_sections,
                "summary_line": summary_line,
                "biopsy": {
                    "positive_systematic": positive_systematic,
                    "positive_targeted": positive_targeted,
                    "percent_sys_pos": percent_sys_pos,
                    "asap_count": asap_count,
                    "epe_count_any": epe_count_any,
                    "pni_count_any": pni_count_any,
                },
            }

            st.toast("Risk assessment saved!", icon="✅")

        if "risk_result" in st.session_state:
            rr = st.session_state["risk_result"]

            st.markdown("### Results")
            col1, col2 = st.columns(2)
            with col1:
                 st.markdown(f"**AJCC Stage:** {rr['ajcc_stage']}")
            with col2:
                 if rr["risk_group"]:
                    st.markdown(f"**NCCN Risk:** {rr['risk_group']}")
                    if rr["subcategory"]:
                         st.caption(f"({rr['subcategory']} intermediate)")
                 else:
                    st.markdown("**NCCN Risk:** Not calculated")

            st.markdown(f"**Disease category:** {rr['disease_category']}")
            st.caption('Localized vs Regional (N1) vs Metastatic (M1). This drives which pathway is shown in Step 3.')

            st.text_area("Copy Summary", value=rr["summary_line"], height=70)

        workflow_nav_buttons(
            back_step=None,
            next_step=2,
            next_disabled=("risk_result" not in st.session_state),
            next_label="Next: Life Expectancy ➡",
        )


    # ============================================================
    # STEP 2: LIFE EXPECTANCY PAGE
    # ============================================================
    elif step == 2:
        st.subheader("Step 2: Life Expectancy Estimation")

        try:
            ssa_df = load_ssa_table()
            ssa_min_age = int(ssa_df["age"].min())
            ssa_max_age = int(ssa_df["age"].max())
        except Exception as e:
            ssa_df = None
            ssa_min_age = None
            ssa_max_age = None
            st.error(f"Could not load SSA table4c6.csv. Error: {e}")

        with st.container(border=True):
            st.markdown("#### Patient Demographics")
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age (years)", min_value=UI_MIN_AGE, max_value=UI_MAX_AGE, value=65, step=1, key="le_age")
            with c2:
                sex = st.selectbox("Sex (life table selection)", options=["Male", "Female"], index=0, key="le_sex")

            adjustment_options = {
                "Best quartile (+50% adjustment)": 1.5,
                "Middle two quartiles (no adjustment)": 1.0,
                "Worst quartile (-50% adjustment)": 0.5,
            }
            health_label = st.selectbox("Overall Health Status", options=list(adjustment_options.keys()), index=1, key="le_health")
            adj_factor = float(adjustment_options[health_label])

        out_of_range = False
        if ssa_df is not None and ssa_min_age is not None and ssa_max_age is not None:
            if int(age) < ssa_min_age or int(age) > ssa_max_age:
                out_of_range = True
                st.warning(f"Age {int(age)} is outside SSA table range available ({ssa_min_age}–{ssa_max_age}).")

        with st.expander("Override with external calculator", expanded=False):
            st.caption("Enable if you prefer to use a value from MSKCC or UCSF calculators.")
            override_on = st.checkbox("Enable manual override", value=False, key="le_override_on")
            override_years = None
            if override_on:
                override_years = st.number_input("Life Expectancy (years)", min_value=0.0, max_value=120.0, value=10.0, step=0.1, key="le_override_years")

        if st.button("Calculate Life Expectancy", type="primary"):
            baseline = None
            adjusted = None

            if ssa_df is None or out_of_range:
                st.info("SSA estimate not available.")
            else:
                baseline = baseline_le_from_ssa(int(age), sex, ssa_df)
                if baseline is None:
                    st.info("No SSA estimate available for this age/sex.")
                else:
                    adjusted = baseline * adj_factor
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown('<div class="le-result-title">SSA Baseline</div>', unsafe_allow_html=True)
                        render_value_block("Years", f"{baseline:.1f}")
                    with c2:
                        st.markdown('<div class="le-result-title">Health Adj.</div>', unsafe_allow_html=True)
                        render_value_block("Factor", f"{adj_factor:.2f}")
                    with c3:
                        st.markdown('<div class="le-result-title">Adjusted Est.</div>', unsafe_allow_html=True)
                        render_value_block("Years", f"{adjusted:.1f}", red=True)

            effective = override_years if override_on else adjusted
            source = "Override" if override_on else "SSA-adjusted"

            st.session_state["le_result"] = {
                "age": int(age),
                "sex": sex,
                "health_label": health_label,
                "adj_factor": adj_factor,
                "baseline_years": baseline,
                "adjusted_years": adjusted,
                "override_on": bool(override_on),
                "override_years": override_years,
                "effective_years": effective,
                "effective_source": source,
                "bucket": le_bucket_from_years(effective),
            }
            st.toast("Life expectancy calculated!", icon="📊")

        st.divider()
        st.markdown("#### External Calculators")
        col_a, col_b = st.columns(2)
        with col_a:
            link_button_or_markdown("MSKCC Male Life Expectancy", MSKCC_LE_URL)
        with col_b:
            link_button_or_markdown("UCSF ePrognosis", UCSF_LEE_URL)

        workflow_nav_buttons(
            back_step=1,
            next_step=3,
            next_disabled=("le_result" not in st.session_state),
            next_label="Next: Pathways ➡",
        )


    # ============================================================
    # STEP 3: EVALUATION & PATHWAYS
    # ============================================================
    elif step == 3:
        st.subheader("Step 3: Clinical Pathways & Evaluation")

        rr = st.session_state.get("risk_result")
        lr = st.session_state.get("le_result")

        if not rr:
            st.error("Missing Risk Data. Please go back to Step 1.")
            workflow_nav_buttons(back_step=1, next_step=None, next_disabled=True)
            st.stop()

        risk_group = rr.get("risk_group")
        subcategory = rr.get("subcategory")
        disease_category = rr.get("disease_category")

        with st.container(border=True):
            st.markdown("#### Clinical Context")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**Disease:** {disease_category}")
            with c2:
                rg_display = risk_group if risk_group else "Not Calculated"
                if subcategory:
                    rg_display += f" ({subcategory})"
                st.markdown(f"**NCCN Risk:** {rg_display}")
            with c3:
                eff = get_effective_le_years()
                eff_str = f"{eff:.1f} years" if eff is not None else "Not set"
                st.markdown(f"**Life Expectancy:** {eff_str}")

        # Additional evaluation summary
        add_title = rr.get("add_title")
        add_items = rr.get("add_items") or []
        if add_title:
            with st.expander("Suggested Evaluation", expanded=False):
                st.info(add_title)
                for it in add_items:
                    st.markdown(f"- {it}")

        st.divider()

        st.subheader("Interactive Pathway")

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            show_tree = st.toggle("Show Decision Tree Diagram", value=True)
        with col_opt2:
            st.checkbox("Patient has symptoms? (Affects >5y branches)", value=False, key="pc_symptoms")

        # Apply pending module navigation BEFORE widget creation
        if "_pending_eval_section" in st.session_state:
            st.session_state["eval_section"] = st.session_state.pop("_pending_eval_section")

        if "eval_section" not in st.session_state:
            st.session_state["eval_section"] = "Initial therapy pathways"

        modules = ["Initial therapy pathways", "Active surveillance program", "Monitoring & recurrence", "Metastatic CSPC"]
        eval_section = st.pills(
            "Select Module",
            modules,
            selection_mode="single",
            default="Initial therapy pathways",
            key="eval_section",
        )

        wiz_key = risk_to_wizard_key(risk_group, subcategory, disease_category)

        def _render_diagram_and_store(spec: WizardSpec):
            wizard_init(spec)
            state = st.session_state[wizard_state_key(spec.id)]
            node = spec.nodes[state["current"]]

            # If current node is auto-routed, highlight the selected edge.
            highlight_edge = None
            handled, _, next_id, edge = _auto_route_for_node(node)
            if handled:
                highlight_edge = edge

            dot = wizard_to_dot(spec, state=state, rankdir="TB", highlight_edge=highlight_edge)
            st.session_state["last_pathway_dot"] = dot
            st.session_state["last_pathway_title"] = spec.title

            # Use container width for better UI
            st.graphviz_chart(dot, use_container_width=True)

        if eval_section == "Initial therapy pathways":
            if disease_category.startswith("Metastatic"):
                st.warning("Disease category is metastatic. Please switch to the **Metastatic CSPC** module.")

            if wiz_key == "low":
                spec = wiz_low_risk("main_low")
                if show_tree:
                     _render_diagram_and_store(spec)
                run_wizard(spec, badges=["Low risk"])

            elif wiz_key == "int_fav":
                spec = wiz_favorable_intermediate("main_int_fav")
                if show_tree:
                    _render_diagram_and_store(spec)
                run_wizard(spec, badges=["Favorable intermediate"])

            elif wiz_key == "int_unfav":
                spec = wiz_unfavorable_intermediate("main_int_unfav")
                if show_tree:
                    _render_diagram_and_store(spec)
                run_wizard(spec, badges=["Unfavorable intermediate"])

            elif wiz_key in {"high", "very_high"}:
                spec = wiz_high_very_high("main_high_vh")
                if show_tree:
                    _render_diagram_and_store(spec)
                run_wizard(spec, badges=["High / Very-high"])

            elif wiz_key == "regional_n1m0":
                spec = wiz_regional_n1m0("main_regional")
                if show_tree:
                    _render_diagram_and_store(spec)
                run_wizard(spec, badges=["N1, M0"])

            else:
                st.info("No wizard key matched. Check risk mapping logic.")

        elif eval_section == "Active surveillance program":
            render_card(
                "Active Surveillance Program",
                "A practical follow-up template.",
                badges=["Follow-up cadence", "Biopsy/MRI", "When to treat"],
            )
            st.markdown(
                """
                <div class="pn-card">
                  <div style="font-size:1.05rem; font-weight:700; margin-bottom:6px;">
                    Active surveillance program — monitoring schedule & key notes
                  </div>
                  <div class="pn-muted">
                    Reference table.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            headers2 = ["Item", "Suggested interval / action", "Notes"]
            rows2: list[list[str]] = []

            rows2.append(["<b>Core follow-up schedule</b>", "", ""])
            rows2.append(["<b>PSA</b>", "Every <b>6 months</b>", "Unless clinically indicated."])
            rows2.append(["<b>DRE</b>", "Every <b>12 months</b>", "Unless clinically indicated."])
            rows2.append(["<b>Repeat biopsy</b>", "Every <b>12 months</b>", "Often every <b>1–3 years</b>."])
            rows2.append(["<b>Repeat MRI</b>", "Consider every <b>12 months</b>", "Unless clinically indicated."])
            rows2.append(["<b>When to leave AS</b>", "Upgrade on biopsy", "Reclassification to GG ≥ 2"])

            body_rows = []
            for r in rows2:
                if r[1] == "" and r[2] == "":
                    body_rows.append(f"<tr class='pn-section-row'><td colspan='3'>{r[0]}</td></tr>")
                else:
                    body_rows.append(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td></tr>")

            html2 = f"<table class='pn-table'><thead><tr>{''.join([f'<th>{h}</th>' for h in headers2])}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
            st.markdown(html2, unsafe_allow_html=True)

        elif eval_section == "Monitoring & recurrence":
            spec = wiz_monitoring_recurrence("monrec_main")
            if show_tree:
                _render_diagram_and_store(spec)
            run_wizard(spec, badges=["Monitoring", "Recurrence"])

        else:
            spec = wiz_metastatic_cspc("m1_main")
            if show_tree:
                _render_diagram_and_store(spec)
            run_wizard(spec, badges=["mCSPC", "Low vs high volume"])

        workflow_nav_buttons(
            back_step=2,
            next_step=None,
            next_disabled=True,
            next_label="",
            extra_right_button=("Generate Report 📄", 4),
        )


    # ============================================================
    # STEP 4: EXPORT REPORT (includes DOT pathway export)
    # ============================================================
    else:
        st.subheader("Step 4: Report Generation")

        rr = st.session_state.get("risk_result")
        lr = st.session_state.get("le_result")

        if not rr:
            st.error("No data found. Please restart.")
            st.stop()

        lines = []
        lines.append(f"{PB_APP_NAME} – Clinical Report")
        lines.append("------------------------------------------------")
        lines.append("PRE-BIOPSY WORKUP (from screening)")
        # PSA history
        hist = st.session_state.get('psa_history', [])
        if isinstance(hist, list) and hist:
            try:
                hist_sorted = sorted(hist, key=lambda x: x.get('date',''))
            except Exception:
                hist_sorted = hist
            for e in hist_sorted:
                d = e.get('date','')
                lbl = e.get('label','PSA')
                v = e.get('psa','')
                try:
                    v_str = f"{float(v):.2f}"
                except Exception:
                    v_str = str(v)
                lines.append(f"- {d}: {v_str} ng/mL ({lbl})")
        else:
            lines.append("- PSA history: not available")

        psad = current_psad()
        if psad is not None:
            lines.append(f"- PSAD: {psad:.2f} ng/mL/cc")

        pctfree = current_free_psa_pct()
        if pctfree is not None:
            lines.append(f"- % Free PSA: {pctfree:.1f}%")

        if bool(st.session_state.get('mri_done', False)):
            lines.append(f"- MRI: {st.session_state.get('pirads','')}")

        bm_entries = st.session_state.get('biomarker_entries', [])
        if isinstance(bm_entries, list) and bm_entries:
            lines.append("- Biomarkers:")
            for e in bm_entries:
                lines.append(f"  • {e.get('date','')}: {e.get('name','')}: {e.get('result','')} ({e.get('interpretation','')})")

        lines.append("")
        lines.append("BIOPSY")
        br = st.session_state.get('biopsy_result','Not done')
        bd = st.session_state.get('biopsy_date', None)
        if br and br != 'Not done':
            if bd:
                lines.append(f"- Date: {bd}")
            lines.append(f"- Result: {br}")
        else:
            lines.append("- Not recorded")

        lines.append("RISK STRATIFICATION")
        lines.append(rr.get("summary_line", ""))
        lines.append(f"AJCC stage: {rr.get('ajcc_stage')}")
        if rr.get("risk_group"):
            rg = rr["risk_group"] + (f" ({rr.get('subcategory')} intermediate)" if rr.get("subcategory") else "")
            lines.append(f"NCCN risk: {rg}")

        lines.append("\nLIFE EXPECTANCY")
        if lr and lr.get("effective_years") is not None:
            lines.append(f"Patient: {lr['age']} year old {lr['sex']}")
            lines.append(f"Effective LE: {lr.get('effective_years')} years")
        else:
            lines.append("Not calculated.")

        lines.append("\nRECOMMENDATIONS & CONSIDERATIONS")
        if rr.get("add_title"):
            lines.append(rr["add_title"])
            for it in rr.get("add_items", []):
                lines.append(f"- {it}")

        if rr.get("treat_title"):
            lines.append("\n" + rr["treat_title"])
            for sec_title, bullets in rr.get("treat_sections", []):
                lines.append(f"\n* {sec_title}")
                for b in bullets:
                    lines.append(f"  - {b}")

        dot = st.session_state.get("last_pathway_dot")

        report_text = "\n".join(lines)

        col1, col2 = st.columns(2)
        with col1:
            st.text_area("Report Preview", value=report_text, height=400)
        with col2:
            st.info("Download Options")

            # PDF DOWNLOAD
            try:
                pdf_bytes = create_pdf(report_text, dot_source=dot)
                st.download_button(
                    "Download Report (.pdf)",
                    data=pdf_bytes,
                    file_name="prostate_report.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating PDF: {e}")

            st.download_button(
                "Download Report (.txt)",
                data=report_text,
                file_name="prostate_report.txt",
                mime="text/plain",
                use_container_width=True
            )

            if dot:
                st.divider()
                st.caption('Diagram Download')
                try:
                    pdf_graph = graphviz.Source(dot).pipe(format='pdf')
                    st.download_button(
                        'Download Pathway Diagram (.pdf)',
                        data=pdf_graph,
                        file_name='pathway.pdf',
                        mime='application/pdf',
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f'Unable to render pathway PDF: {e}')

        workflow_nav_buttons(back_step=3, next_step=None, next_disabled=True)

    st.markdown(
        f"""
        <div style="text-align:center; font-size:0.8rem; color:#888; padding-top:2rem;">
            © {PB_APP_YEAR} {PB_APP_AUTHOR} | For educational use only.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# COMBINED MAIN ROUTER
# ============================================================
init_state()
_sticky_restore()
_sticky_update()
_psa_persistence_tick()
qp = get_qp()
if qp_first(qp, 'focus', '0') == '1':
    focus_mode()
    st.stop()

step = int(st.session_state.wizard_step)
if step < 5:
    progress_header()
    sidebar_tracker()
    if step == 1: page_screening()
    elif step == 2: page_evaluation()
    elif step == 3: page_biopsy_result()
    elif step == 4: page_after_negative_biopsy()
else:
    entering = bool(st.session_state.pop("_enter_post_biopsy", False))

    # PSA handoff (Screening → Post-biopsy)
    # Set the post-biopsy PSA widget key BEFORE the widget is created.
    # We intentionally do not pass value= to the widget to avoid Streamlit warnings.
    if entering:
        psa_from_screen = _current_psa_from_screening()
        if psa_from_screen is not None:
            try:
                st.session_state[rk("psa")] = float(psa_from_screen)
            except Exception:
                pass

    prefill_post_biopsy_from_screening(force=entering)
    post_biopsy_main()

