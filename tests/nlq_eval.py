"""Run the NLQ evaluation set and print accuracy by group.   python tests/nlq_eval.py   (add -v to list failures)"""
import sys, time, pathlib, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])); warnings.filterwarnings("ignore")
from collections import defaultdict

from engine.data_loader import load_master
from nlp.chatbot import answer
from tests.nlq_cases import CASES, CASES_B, CASES_C

KEYS = ("category", "year", "year_from", "year_to", "year_a", "year_b", "top_n", "state_a", "state_b", "state_filter", "x", "y", "bottom", "exact_only")


def check(df, case):
    group, text, qid, exp, ctx = case
    try:
        r = answer(df, text, context=({"category": "Overall", **(ctx or {})}), use_gemini=False)
    except Exception as e:  # an exception is always a failure
        return False, False, f"EXC {e!r}"[:90]
    if qid is None:
        ok = (not r.ok) and ("msg" not in exp or exp["msg"].lower() in r.message.lower())
        return ok, ok, "" if ok else f"expected decline, got ok={r.ok} {getattr(r.result,'qid',None)}"
    if not r.ok:
        return False, False, f"declined: {r.message[:70]}"
    q_ok = r.result.qid == qid
    p, why = r.result.params, []
    for k, v in exp.items():
        if k in ("inst", "inst_b"):
            key = p.get("institution" if k == "inst" else "institution_b")
            name = df.loc[df.institution_key.eq(key), "institute_name"].iloc[0] if key else ""
            if v.lower() not in name.lower(): why.append(f"{k}={name!r}")
        elif k in KEYS and p.get(k) != v:
            why.append(f"{k}={p.get(k)!r}≠{v!r}")
    return q_ok, q_ok and not why, (f"got {r.result.qid}" if not q_ok else "") + (" " + ", ".join(why) if why else "")


def run(verbose=False, cases=None):
    df = load_master(); g = defaultdict(lambda: [0, 0, 0]); fails = []; t0 = time.time()
    for c in (cases or CASES):
        q, full, note = check(df, c)
        s = g[c[0]]; s[0] += 1; s[1] += q; s[2] += full
        if not full: fails.append((c[0], c[1], note))
    n = sum(v[0] for v in g.values()); q = sum(v[1] for v in g.values()); f = sum(v[2] for v in g.values())
    print(f"{'group':12} {'n':>3} {'intent':>8} {'intent+params':>14}")
    for k, v in g.items(): print(f"{k:12} {v[0]:>3} {100*v[1]/v[0]:>7.0f}% {100*v[2]/v[0]:>13.0f}%")
    print(f"{'TOTAL':12} {n:>3} {100*q/n:>7.0f}% {100*f/n:>13.0f}%   ({(time.time()-t0)/n*1000:.0f} ms/query)")
    if verbose:
        for grp, t, note in fails: print(f"  [{grp}] {t!r} -> {note}")
    return q / n, f / n


if __name__ == "__main__":
    run("-v" in sys.argv)
    print("\n--- second set (first run: 79% intent; tuned since) ---")
    run("-v" in sys.argv, CASES_B)
    print("\n--- third set (first run: 66% intent; tuned since) ---")
    run("-v" in sys.argv, CASES_C)
