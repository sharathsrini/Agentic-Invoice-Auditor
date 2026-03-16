# Run: streamlit run ai_invoice_auditor/streamlit_app.py
"""AI Invoice Auditor -- Streamlit UI.

Three-tab interface:
    1. Live Audit   -- View processed invoice statuses (XCUT-03)
    2. Human Review -- Placeholder for Plan 03 (XCUT-04/05)
    3. RAG Q&A      -- Query invoices via RAG pipeline (XCUT-06)

IMPORTANT: This module communicates with FastAPI exclusively via HTTP requests.
No internal ai_invoice_auditor module imports are used.
"""

import streamlit as st
import requests
import pandas as pd

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="AI Invoice Auditor", layout="wide")
st.title("AI Invoice Auditor")

tab_audit, tab_review, tab_rag = st.tabs(["Live Audit", "Human Review", "RAG Q&A"])

# ---------------------------------------------------------------------------
# Tab 1: Live Audit
# ---------------------------------------------------------------------------
with tab_audit:
    st.header("Live Audit Status")

    if st.button("Refresh Status", key="refresh_audit"):
        st.rerun()

    try:
        resp = requests.get(f"{API_BASE}/invoice/", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            invoices = data.get("invoices", [])

            st.metric("Total Invoices", len(invoices))

            if invoices:
                # Build display dataframe
                df = pd.DataFrame(invoices)

                # Add status indicator column
                status_icons = {
                    "auto_approve": "🟢",
                    "manual_review": "🟠",
                    "interrupted": "🟠",
                    "reject": "🔴",
                    "flag": "🟡",
                    "unknown": "⚪",
                }
                df["Status Indicator"] = df["status"].map(
                    lambda s: status_icons.get(s, "⚪")
                )

                # Reorder columns for display
                display_cols = ["invoice_no", "Status Indicator", "status", "report_path"]
                display_cols = [c for c in display_cols if c in df.columns]
                st.dataframe(df[display_cols], use_container_width=True)
            else:
                st.info("No invoices processed yet. Submit an invoice via the API.")
        else:
            st.error(f"API returned status {resp.status_code}: {resp.text}")
    except requests.ConnectionError:
        st.error("Cannot connect to API. Is the FastAPI server running?")
    except requests.Timeout:
        st.error("API request timed out.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# ---------------------------------------------------------------------------
# Tab 2: Human Review (XCUT-04 + XCUT-05)
# ---------------------------------------------------------------------------


def _build_corrections(edited_df: pd.DataFrame, original_df: pd.DataFrame) -> dict:
    """Compare edited DataFrame against original and return dict of changed values.

    The business_validator_agent resume handler expects:
        {"decision": "edit", "corrections": {"field_name": "new_value", ...}}
    where field_name maps to keys in the extracted dict. We iterate edited rows,
    detect changes in invoice_value, and build the corrections dict keyed by
    each row's field name.
    """
    corrections: dict = {}
    for idx in range(len(edited_df)):
        edited_val = edited_df.iloc[idx].get("invoice_value")
        original_val = original_df.iloc[idx].get("invoice_value")
        if edited_val != original_val:
            field_name = edited_df.iloc[idx].get("field", f"field_{idx}")
            corrections[field_name] = edited_val
    return corrections


with tab_review:
    st.header("Human Review")

    # 1. Fetch invoices needing review
    try:
        resp_list = requests.get(f"{API_BASE}/invoice/", timeout=5)
        if resp_list.status_code == 200:
            all_invoices = resp_list.json().get("invoices", [])
            review_invoices = [
                inv
                for inv in all_invoices
                if inv.get("status") in ("interrupted", "manual_review")
            ]
        else:
            review_invoices = []
            st.error(f"API returned status {resp_list.status_code}: {resp_list.text}")
    except requests.ConnectionError:
        review_invoices = []
        st.error("Cannot connect to API. Is the FastAPI server running?")
    except Exception as e:
        review_invoices = []
        st.error(f"Unexpected error fetching invoices: {e}")

    if not review_invoices:
        st.info("No invoices currently require human review.")
    else:
        # 2. Invoice selector
        invoice_options = [inv["invoice_no"] for inv in review_invoices]
        selected_invoice = st.selectbox(
            "Select invoice for review:",
            options=invoice_options,
            key="review_select",
        )

        # Fetch full invoice details
        invoice_data = None
        try:
            resp_detail = requests.get(
                f"{API_BASE}/invoice/{selected_invoice}", timeout=5
            )
            if resp_detail.status_code == 200:
                invoice_data = resp_detail.json()
            else:
                st.error(f"Failed to load invoice details: {resp_detail.text}")
        except requests.ConnectionError:
            st.error("Cannot connect to API. Is the FastAPI server running?")
        except Exception as e:
            st.error(f"Unexpected error loading invoice: {e}")

        if invoice_data is not None:
            # 3. Display invoice summary
            extracted = invoice_data.get("extracted") or {}
            st.subheader("Invoice Summary")
            summary_cols = st.columns(5)
            with summary_cols[0]:
                st.write(f"**Invoice No:** {extracted.get('invoice_no', 'N/A')}")
            with summary_cols[1]:
                st.write(f"**Vendor ID:** {extracted.get('vendor_id', 'N/A')}")
            with summary_cols[2]:
                st.write(f"**PO Number:** {extracted.get('po_number', 'N/A')}")
            with summary_cols[3]:
                st.write(f"**Total:** {extracted.get('total_amount', 'N/A')}")
            with summary_cols[4]:
                st.write(f"**Currency:** {extracted.get('currency', 'N/A')}")

            st.warning(
                f"Status: {invoice_data.get('status', 'unknown')} "
                f"-- This invoice requires human review"
            )

            # 4. Display discrepancies as editable table (XCUT-04)
            validation_result = invoice_data.get("validation_result") or {}
            discrepancies = validation_result.get("discrepancies", [])

            if discrepancies:
                st.subheader("Discrepancies")
                original_df = pd.DataFrame(discrepancies)
                edited_df = st.data_editor(
                    original_df,
                    key="disc_editor",
                    num_rows="fixed",
                    disabled=["item_code", "field", "erp_value", "deviation_pct"],
                )
            else:
                st.info("No discrepancies found in validation result.")
                original_df = pd.DataFrame()
                edited_df = pd.DataFrame()

            # 5. Action buttons (XCUT-05)
            st.subheader("Actions")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Approve", key="btn_approve", type="primary"):
                    try:
                        resp_action = requests.post(
                            f"{API_BASE}/invoice/{selected_invoice}/resume",
                            json={"decision": "approve"},
                            timeout=30,
                        )
                        if resp_action.status_code == 200:
                            st.success("Invoice approved. Pipeline resumed.")
                        else:
                            st.error(f"Action failed: {resp_action.text}")
                    except requests.ConnectionError:
                        st.error(
                            "Cannot connect to API. Is the FastAPI server running?"
                        )
                    except Exception as e:
                        st.error(f"Action failed: {e}")

            with col2:
                if st.button("Reject", key="btn_reject"):
                    try:
                        resp_action = requests.post(
                            f"{API_BASE}/invoice/{selected_invoice}/resume",
                            json={"decision": "reject"},
                            timeout=30,
                        )
                        if resp_action.status_code == 200:
                            st.success("Invoice rejected.")
                        else:
                            st.error(f"Action failed: {resp_action.text}")
                    except requests.ConnectionError:
                        st.error(
                            "Cannot connect to API. Is the FastAPI server running?"
                        )
                    except Exception as e:
                        st.error(f"Action failed: {e}")

            with col3:
                if st.button("Apply Edits", key="btn_edit"):
                    if discrepancies:
                        corrections = _build_corrections(edited_df, original_df)
                        if not corrections:
                            st.warning(
                                "No edits detected. Modify invoice_value fields "
                                "in the table above first."
                            )
                        else:
                            try:
                                resp_action = requests.post(
                                    f"{API_BASE}/invoice/{selected_invoice}/resume",
                                    json={
                                        "decision": "edit",
                                        "corrections": corrections,
                                    },
                                    timeout=30,
                                )
                                if resp_action.status_code == 200:
                                    st.success(
                                        "Edits applied. Pipeline re-running "
                                        "from validation."
                                    )
                                else:
                                    st.error(f"Action failed: {resp_action.text}")
                            except requests.ConnectionError:
                                st.error(
                                    "Cannot connect to API. "
                                    "Is the FastAPI server running?"
                                )
                            except Exception as e:
                                st.error(f"Action failed: {e}")
                    else:
                        st.warning("No discrepancies to edit.")

# ---------------------------------------------------------------------------
# Tab 3: RAG Q&A
# ---------------------------------------------------------------------------
with tab_rag:
    st.header("RAG Q&A")
    st.write("Ask questions about your processed invoices.")

    query = st.text_input("Ask about your invoices:", key="rag_query")

    if st.button("Search", key="rag_search"):
        if not query or not query.strip():
            st.warning("Please enter a query.")
        else:
            try:
                resp = requests.post(
                    f"{API_BASE}/rag/query",
                    json={"query": query},
                    timeout=30,
                )

                if resp.status_code == 200:
                    data = resp.json()

                    # Answer
                    st.subheader("Answer")
                    st.write(data.get("answer", "No answer returned."))

                    # Quality Scores (RAG Triad)
                    triad_scores = data.get("triad_scores", {})
                    if triad_scores:
                        st.subheader("Quality Scores")
                        for metric_name, score in triad_scores.items():
                            col_label, col_badge = st.columns([3, 1])
                            with col_label:
                                st.write(metric_name.replace("_", " ").title())
                            with col_badge:
                                score_val = float(score) if score is not None else 0.0
                                color = "🟢" if score_val >= 0.7 else "🔴"
                                st.markdown(f"{color} **{score_val:.2f}**")

                    # Source Chunks
                    source_chunks = data.get("source_chunks", [])
                    if source_chunks:
                        st.subheader("Source Chunks")
                        for i, chunk in enumerate(source_chunks):
                            chunk_text = chunk.get("text", str(chunk))
                            chunk_score = chunk.get("score", "N/A")
                            with st.expander(f"Chunk {i + 1} (score: {chunk_score})"):
                                st.write(chunk_text)
                else:
                    st.error(f"Query failed: {resp.text}")

            except requests.ConnectionError:
                st.error("Cannot connect to API. Is the FastAPI server running?")
            except requests.Timeout:
                st.error("API request timed out.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
