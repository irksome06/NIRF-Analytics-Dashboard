"""Ask NIRF: natural-language interface. Rules first, Gemini only for unclear questions, always validated.
Remembers the previous question so follow-ups work ("what about engineering?", "show more", "now 2020")."""
from __future__ import annotations

import datetime
import pandas as pd
import streamlit as st
import os

from app import ui
from engine.registry import SECTION_TITLES, by_section
from nlp.chatbot import EXAMPLES, TOPICS, answer
from nlp.gemini import get_api_key


def _render_export_markdown(hist: list[dict]) -> str:
    """Generate a clean Markdown audit report of the conversation."""
    lines = [
        "# NIRF Dashboard — Chat Session Report",
        f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "---",
    ]
    for i, e in enumerate(hist):
        role = "User" if e["role"] == "user" else "Assistant"
        lines.append(f"### {i+1}. {role}\n")
        lines.append(e.get("text", "") + "\n")
        if e.get("metrics"):
            lines.append("**Key Metrics:**")
            for m in e["metrics"]:
                lines.append(f"- **{m['label']}:** {m['value']}")
            lines.append("")
        if e.get("insights"):
            lines.append("**Key Insights:**")
            for b in e["insights"]:
                lines.append(f"- {b}")
            lines.append("")
    return "\n".join(lines)


def _render(entry: dict, i: int) -> None:
    with st.chat_message(entry["role"], avatar=":material/person:" if entry["role"] == "user" else ":material/query_stats:"):
        st.markdown(entry["text"])

        # Disambiguation buttons (e.g. if user typed an ambiguous term like "IIT")
        disambig = entry.get("disambiguation", [])
        if disambig:
            st.info("💡 **Multiple institutions matched your query.** Select one below:")
            d_cols = st.columns(min(len(disambig), 4))
            for k, inst_name in enumerate(disambig[:4]):
                if d_cols[k].button(inst_name, key=f"disambig_{i}_{k}", width="stretch"):
                    st.session_state["pending_prompt"] = f"ranking trend of {inst_name}"
                    st.rerun()

        # Engine Badges (Latency + Origin + QID)
        source = entry.get("source")
        qid = entry.get("qid")
        latency = entry.get("latency_ms", 0)
        if source:
            src_label = "⚡ Local NLP Engine" if source == "rules" else "🤖 Gemini Fallback"
            src_cls = "chat-badge-rules" if source == "rules" else "chat-badge-gemini"
            badges = f"<span class='chat-badge {src_cls}'>{src_label} ({latency:.0f}ms)</span>"
            if qid:
                badges += f"<span class='chat-badge chat-badge-meta'>Question {qid}</span>"
            st.markdown(badges, unsafe_allow_html=True)

        # KPI Metric Strip
        metrics = entry.get("metrics", [])
        if metrics:
            m_cols = st.columns(min(len(metrics), 4))
            for mc, m in zip(m_cols, metrics[:4]):
                ui.kpi_card(mc, m["label"], m["value"], m.get("delta"), m.get("help"))

        # Key Findings / Insights Box
        insights = entry.get("insights", [])
        if insights:
            st.markdown(
                "<div class='chat-insights-card'><h5>📌 Key Takeaways</h5><ul>" +
                "".join(f"<li>{ui._md_bold(b)}</li>" for b in insights) +
                "</ul></div>",
                unsafe_allow_html=True
            )
            audio_text = "Key findings: " + ". ".join(b.replace("**", "").replace("*", "") for b in insights)
            ui.render_speech_button(audio_text, f"chat_{i}")

        res = entry.get("result")
        if res is not None:
            st.plotly_chart(res.figure, width="stretch", key=f"chat_fig_{i}", config={"displaylogo": False})
            
            # Action toolbar: CSV export + Full data view
            tb_col1, tb_col2 = st.columns([1, 4])
            if hasattr(res, "data") and isinstance(res.data, pd.DataFrame) and not res.data.empty:
                csv_bytes = res.data.to_csv(index=False).encode("utf-8")
                tb_col1.download_button(
                    label="📥 Download CSV",
                    data=csv_bytes,
                    file_name=f"nirf_{res.qid}_{entry.get('category', 'data')}.csv",
                    mime="text/csv",
                    key=f"dl_csv_{i}"
                )
            with st.expander(f"Data ({len(res.data):,} rows)"):
                st.dataframe(res.data, hide_index=True, width="stretch")


def _chips(prompts: list[str], key: str, label: str) -> None:
    if not prompts:
        return
    st.caption(label)
    cols = st.columns(min(len(prompts), 3))
    for j, p in enumerate(prompts[:3]):
        if cols[j].button(p, key=f"{key}_{j}", width="stretch"):
            st.session_state["pending_prompt"] = p
            st.rerun()


def page() -> None:
    df = ui.load_data()
    flt = ui.get_filters(df)
    ui.page_header("Ask NIRF", "Ask in plain English. The assistant picks the matching analysis or builds a custom chart, checks every filter against the data, and draws it. "
                              "It remembers your last question, so you can follow up with “what about engineering?” or “show more”.")
    pool = df.drop_duplicates("institution_key")
    hist = st.session_state.setdefault("chat_history", [])

    # Status Bar
    has_gemini = bool(get_api_key())
    gemini_status = "⚡ Gemini Fallback: **Active**" if has_gemini else "⚡ Gemini Fallback: **Standby** (Add GEMINI_API_KEY to enable; local engine active)"
    st.caption(f"{gemini_status} · Category context: **{flt['category']}** · Years: **2017–2025**")
    # Answer source selector
    source_choice = st.radio(
        "Answer source",
        ["Local model", "Gemini"],
        index=1 if has_gemini else 0,
        key="answer_source",
    )

    # Expandable Annexure A Question Explorer
    with st.expander("📚 Browse All 40 Annexure A Questions", expanded=False):
        st.markdown("Select an evaluation category to view and launch any official question directly into chat:")
        sec_choice = st.selectbox(
            "Evaluation Section",
            options=list(SECTION_TITLES.keys()),
            format_func=lambda s: f"Section {s}: {SECTION_TITLES[s]}",
            key="chat_sec_browser"
        )
        sec_questions = by_section(sec_choice)
        q_cols = st.columns(min(len(sec_questions), 2))
        for idx, q in enumerate(sec_questions):
            col_target = q_cols[idx % 2]
            btn_label = f"Q{q.qid}: {q.title}"
            prompt_text = q.title
            if "institution" in q.params and not any(k in prompt_text for k in ["IIT", "IISc"]):
                prompt_text = f"{q.title.rstrip('?')} for IIT Madras?"
            if col_target.button(btn_label, key=f"annex_btn_{q.qid}", width="stretch"):
                st.session_state["pending_prompt"] = prompt_text
                st.rerun()

    prompt = st.session_state.pop("pending_prompt", None)
    typed = st.chat_input("Ask about NIRF rankings, e.g. “Top 10 engineering colleges in 2024”")
    prompt = prompt or typed
    if not hist:
        st.markdown("**Try one of these**")
        tabs = st.tabs(list(TOPICS))
        for t, (topic, items) in zip(tabs, TOPICS.items()):
            with t:
                for k, ex in enumerate(items):
                    if st.button(ex, key=f"ex_{topic}_{k}", width="stretch"):
                        prompt = ex
    if prompt:
        hist.append({"role": "user", "text": prompt})
        # Determine which engine to use based on user selection
        use_gemini = source_choice == "Gemini"
        original_key = os.getenv("GEMINI_API_KEY")
        if not use_gemini and "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]
        try:
            with st.spinner("Analyzing NIRF records…"):
                rep = answer(df, prompt, context={"category": flt["category"], "last_query": st.session_state.get("nirf_last_query")}, use_gemini=use_gemini, pool=pool)

        finally:
            # Restore original API key if it existed
            if original_key is not None:
                os.environ["GEMINI_API_KEY"] = original_key
            elif "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
        if rep.ok:
            st.session_state["nirf_last_query"] = rep.resolved
        entry = {
            "role": "assistant",
            "text": rep.message,
            "result": rep.result if rep.ok else None,
            "alts": [p for _, p in rep.alternatives],
            "next": rep.suggestions,
            "source": rep.source,
            "qid": getattr(rep.result, "qid", None) if rep.ok else None,
            "category": rep.resolved.get("category", flt["category"]) if rep.ok else flt["category"],
            "insights": rep.insights,
            "metrics": rep.metrics,
            "disambiguation": rep.disambiguation,
            "latency_ms": rep.latency_ms,
        }
        if not rep.ok and not rep.disambiguation:
            entry["text"] += "\n\nTry: " + " · ".join(f"*{e}*" for e in EXAMPLES[:3])
        hist.append(entry)

    for i, e in enumerate(hist):
        _render(e, i)

    if hist and hist[-1]["role"] == "assistant":
        last = hist[-1]
        _chips(last.get("alts", []), f"alt_{len(hist)}", "Not what you meant? Try:")
        _chips(last.get("next", []), f"next_{len(hist)}", "Keep exploring:")

    # Chat utilities (Clear and Export)
    if hist:
        st.divider()
        ctrl_col1, ctrl_col2, _ = st.columns([1, 1.2, 3])
        if ctrl_col1.button("Clear conversation", key="chat_clear"):
            st.session_state["chat_history"] = []
            st.session_state.pop("nirf_last_query", None)
            st.rerun()
        report_md = _render_export_markdown(hist)
        ctrl_col2.download_button(
            label="📄 Export Chat Report",
            data=report_md.encode("utf-8"),
            file_name=f"nirf_chat_report_{datetime.date.today().isoformat()}.md",
            mime="text/markdown",
            key="chat_export_report"
        )
