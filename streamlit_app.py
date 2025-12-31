from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import yaml

from mcqgen.logging_utils import setup_run_logger
from mcqgen.pipeline import run_generate_pipeline
from mcqgen.explain import add_explanations_for_wrong_questions
from mcqgen.llm_client import make_client
from mcqgen.run_utils import make_run_id

CONTEXTS_DIR = Path("./contexts")
RUNS_DIR = Path("./runs")
CONFIGS_DIR = Path("./configs")
PROMPTS_DIR = Path("./prompts")

st.set_page_config(page_title="MCQGen", layout="wide")
st.title("MCQGen")


def list_pdfs() -> list[str]:
    CONTEXTS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted([p.name for p in CONTEXTS_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_output(run_dir: Path) -> dict:
    return json.loads((run_dir / "output.json").read_text(encoding="utf-8"))


def letters_for_choices(choices: list[dict]) -> dict:
    id_to_letter: dict[str, str] = {}
    for idx, c in enumerate(choices):
        cid = c.get("id", "")
        if cid:
            id_to_letter[cid] = chr(ord("A") + idx)
    return id_to_letter


def list_prompt_files() -> list[str]:
    if not PROMPTS_DIR.exists():
        return []
    files = sorted([p for p in PROMPTS_DIR.glob("*.txt") if p.is_file()])
    files = [p for p in files if p.name != "explain_v1.txt"]
    return [str(p.relative_to(Path("."))).replace("\\", "/") for p in files]


# ---------- Sidebar: Context ----------
st.sidebar.header("1) Context")
pdfs = list_pdfs()

uploaded = st.sidebar.file_uploader("Upload PDF (optional)", type=["pdf"])
if uploaded is not None:
    out_path = CONTEXTS_DIR / uploaded.name
    out_path.write_bytes(uploaded.getvalue())
    st.sidebar.success(f"Saved to contexts/: {uploaded.name}")
    pdfs = list_pdfs()

context_name = st.sidebar.selectbox("Select context PDF", pdfs if pdfs else ["(no PDFs found)"])
context_pdf = CONTEXTS_DIR / context_name if pdfs else None

# ---------- Sidebar: Config ----------
st.sidebar.header("2) Settings")

config_files = sorted([p.name for p in CONFIGS_DIR.iterdir() if p.suffix in [".yml", ".yaml"]]) if CONFIGS_DIR.exists() else []
config_name = st.sidebar.selectbox("Base config file", config_files if config_files else ["(no configs found)"])
cfg = load_cfg(CONFIGS_DIR / config_name) if config_files else None

use_mock_llm = False

if cfg:
    qt = st.sidebar.selectbox("Question type", ["MCQ", "SATA"], index=0 if cfg["generation"]["question_type"] == "MCQ" else 1)
    choices = st.sidebar.number_input("Choices per question", 2, 8, int(cfg["generation"]["choices_per_question"]))
    pages_per = st.sidebar.number_input("Pages per partition", 1, 50, int(cfg["partitioning"]["pages_per_partition"]))
    overlap = st.sidebar.number_input("Overlap pages", 0, int(pages_per) - 1, int(cfg["partitioning"]["overlap_pages"]))
    q_per_part = st.sidebar.number_input("Questions per partition", 1, 50, int(cfg["generation"]["questions_per_partition"]))

    rand_q = st.sidebar.toggle("Randomize question order", value=bool(cfg["randomization"]["randomize_questions"]))
    rand_opt = st.sidebar.toggle("Randomize answer options", value=bool(cfg["randomization"]["randomize_options"]))

    prompt_files = list_prompt_files()
    prompt_default = cfg["prompts"]["user_prompt_file"].replace("\\", "/")
    prompt_idx = prompt_files.index(prompt_default) if prompt_default in prompt_files else 0
    generator_prompt = st.sidebar.selectbox("Generator prompt file", prompt_files, index=prompt_idx)

    cfg["generation"]["question_type"] = qt
    cfg["generation"]["choices_per_question"] = int(choices)
    cfg["generation"]["questions_per_partition"] = int(q_per_part)
    cfg["partitioning"]["pages_per_partition"] = int(pages_per)
    cfg["partitioning"]["overlap_pages"] = int(overlap)
    cfg["randomization"]["randomize_questions"] = bool(rand_q)
    cfg["randomization"]["randomize_options"] = bool(rand_opt)
    if prompt_files:
        cfg["prompts"]["user_prompt_file"] = generator_prompt

# ---------- Main: Generate ----------
st.header("3) Generate Questions")

if "run_dir" not in st.session_state:
    st.session_state.run_dir = None
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "wrong_ids" not in st.session_state:
    st.session_state.wrong_ids = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "active_run_name" not in st.session_state:
    st.session_state.active_run_name = None
if "explanations_ready" not in st.session_state:
    st.session_state.explanations_ready = False

col1, col2 = st.columns([1, 2])

with col1:
    generate_clicked = st.button("Generate", type="primary", disabled=(context_pdf is None or cfg is None))

with col2:
    if st.session_state.run_dir:
        st.success(f"Current run: {st.session_state.run_dir.name}")

if generate_clicked:
    st.info("Generating... check runs/<run>/run.log if something fails.")
    try:
        run_id = make_run_id(context_pdf)
        run_dir = RUNS_DIR / run_id
        logger = setup_run_logger(run_dir, verbose=True)
        progress = st.progress(0.0)

        def on_progress(current: int, total: int) -> None:
            if total:
                progress.progress(current / total)

        run_dir = run_generate_pipeline(
            context_pdf=context_pdf,
            cfg=cfg,
            runs_dir=RUNS_DIR,
            logger=logger,
            use_mock_llm=use_mock_llm,
            run_dir=run_dir,
            config_path=CONFIGS_DIR / config_name if config_files else None,
            progress_callback=on_progress
        )

        st.session_state.run_dir = run_dir
        st.session_state.answers = {}
        st.session_state.wrong_ids = []
        st.session_state.current_index = 0
        st.session_state.submitted = False
        st.session_state.active_run_name = run_dir.name
        st.session_state.explanations_ready = False
        st.success("Generation complete.")
    except Exception as exc:
        st.error(f"Generation failed: {exc}")

# ---------- Main: Test UI ----------
st.header("4) Tests")

available_runs = sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()]) if RUNS_DIR.exists() else []
run_options = ["(use current run)"] + [p.name for p in available_runs]
selected_run = st.selectbox("Load an existing run", run_options)
if selected_run != "(use current run)":
    st.session_state.run_dir = RUNS_DIR / selected_run
if st.session_state.active_run_name != selected_run:
    st.session_state.answers = {}
    st.session_state.wrong_ids = []
    st.session_state.current_index = 0
    st.session_state.submitted = False
    st.session_state.active_run_name = selected_run
    st.session_state.explanations_ready = False

if st.session_state.run_dir and (st.session_state.run_dir / "output.json").exists():
    output = load_output(st.session_state.run_dir)
    questions = output["questions"]
    qtype = output["settings"]["question_type"]

    total_q = len(questions)
    idx = max(0, min(st.session_state.current_index, total_q - 1))
    q = questions[idx]
    choices_map = {c["id"]: c["text"] for c in q["choices"]}
    id_to_letter = letters_for_choices(q["choices"])

    try:
        q_num = int(str(q["id"]).split("_")[1])
    except Exception:
        q_num = idx + 1

    st.subheader("Answer questions")
    st.markdown(f"**Question {q_num}** - {q['stem']}")
    st.caption(f"Question {idx + 1} / {total_q} | Solved: {idx} | Remaining: {total_q - idx}")

    if not st.session_state.submitted:
        with st.form(key=f"qform_{q['id']}"):
            if qtype == "MCQ":
                selected = st.radio(
                    label="",
                    options=list(choices_map.keys()),
                    index=None,
                    format_func=lambda cid: f"{id_to_letter.get(cid, '?')}. {choices_map[cid]}",
                    key=f"ans_{q['id']}"
                )
                st.session_state.answers[q["id"]] = [selected] if selected else []
            else:
                selected_ids = []
                for cid in choices_map.keys():
                    label = f"{id_to_letter.get(cid, '?')}. {choices_map[cid]}"
                    checked = st.checkbox(label, key=f"ans_{q['id']}_{cid}")
                    if checked:
                        selected_ids.append(cid)
                st.session_state.answers[q["id"]] = selected_ids

            button_label = "Next" if idx < total_q - 1 else "Submit Test"
            submitted_step = st.form_submit_button(button_label)

        if submitted_step:
            current_selected = st.session_state.answers.get(q["id"], [])
            if len(current_selected) == 0:
                st.warning("Please select at least one option to continue.")
            elif idx < total_q - 1:
                st.session_state.current_index += 1
                st.rerun()
            else:
                wrong = []
                correct_count = 0
                for q_item in questions:
                    got = set(st.session_state.answers.get(q_item["id"], []))
                    correct = set(q_item["correct_choice_ids"])
                    if got == correct:
                        correct_count += 1
                    else:
                        wrong.append(q_item["id"])
                st.session_state.wrong_ids = wrong
                st.session_state.submitted = True
                st.rerun()

    if st.session_state.submitted:
        total = len(questions)
        correct_count = total - len(st.session_state.wrong_ids)
        percent = round((correct_count / total) * 100, 2) if total else 0.0
        st.subheader("Results")
        st.info(f"Score: {correct_count} / {total} ({percent}%)")

        output = load_output(st.session_state.run_dir)
        by_id = {q_item["id"]: q_item for q_item in output["questions"]}
        for qid in [q_item["id"] for q_item in questions]:
            q_item = by_id.get(qid)
            if not q_item:
                continue
            user_sel = set(st.session_state.answers.get(qid, []))
            correct = set(q_item["correct_choice_ids"])
            is_correct = user_sel == correct
            status = "✅" if is_correct else "❌"

            try:
                q_num = int(str(qid).split("_")[1])
            except Exception:
                q_num = None
            label = f"Question {q_num}" if q_num is not None else "Question"
            st.markdown(f"{status} **{label}** - {q_item['stem']}")

            id_to_letter = letters_for_choices(q_item["choices"])
            for c in q_item["choices"]:
                cid = c["id"]
                text = c["text"]
                letter = id_to_letter.get(cid, "?")
                selected = cid in user_sel

                if qtype == "MCQ":
                    if cid in user_sel and cid in correct:
                        marker = "✅"
                    elif cid in user_sel and cid not in correct:
                        marker = "❌"
                    elif cid in correct:
                        marker = "✅"
                    else:
                        marker = " "
                else:
                    if (cid in correct and cid in user_sel) or (cid not in correct and cid not in user_sel):
                        marker = "✅"
                    else:
                        marker = "❌"

                cols = st.columns([0.015, 0.965], gap="small")
                with cols[0]:
                    st.checkbox(
                        label="",
                        value=selected,
                        disabled=True,
                        key=f"res_{qid}_{cid}"
                    )
                with cols[1]:
                    st.markdown(
                        f"<div style='margin-top: 10px'>{marker} {letter}. {text}</div>",
                        unsafe_allow_html=True
                    )

            correct_letters = sorted([id_to_letter.get(cid, "?") for cid in correct])
            user_letters = sorted([id_to_letter.get(cid, "?") for cid in user_sel])
            st.caption(f"Correct: {', '.join(correct_letters)} | Your answer: {', '.join(user_letters)}")
            if not is_correct:
                if q_item.get("explanation"):
                    st.markdown(
                        f"**GPT Explanation**\n\n{q_item['explanation']}",
                    )
                else:
                    st.caption("GPT Explanation not available yet.")
            st.divider()

        if st.session_state.wrong_ids:
            if st.button("Explain wrong answers (GPT)", disabled=use_mock_llm):
                try:
                    with st.spinner("Generating explanations..."):
                        client = make_client()
                        manifest = json.loads((st.session_state.run_dir / "manifest.json").read_text(encoding="utf-8"))
                        explain_prompt = Path(manifest["prompt_files"]["explanation"])
                        model = manifest["llm"]["model"]

                        updated = add_explanations_for_wrong_questions(
                            output_json_path=st.session_state.run_dir / "output.json",
                            wrong_ids=st.session_state.wrong_ids,
                            explain_prompt_file=explain_prompt,
                            client=client,
                            model=model,
                            reasoning_effort=manifest["llm"].get("reasoning_effort", "none"),
                            temperature=manifest["llm"].get("temperature", 0.2),
                            max_output_tokens=600,
                            logger=None
                        )
                        (st.session_state.run_dir / "output.json").write_text(
                            json.dumps(updated, indent=2, ensure_ascii=False),
                            encoding="utf-8"
                        )
                        st.session_state.explanations_ready = True
                    st.success("Explanations added.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Explain failed: {exc}")
else:
    st.caption("Generate a run first to answer questions here.")
