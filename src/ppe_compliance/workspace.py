"""Presentation components. Importing this module does not load YOLO or Streamlit."""

from html import escape
import json


PIPELINE = (
    ('01', 'Validate the source', 'Decode image or video pixels. Check dimensions, confidence and file readability before inference.'),
    ('02', 'Detect three classes', 'YOLO11 nano predicts person, helmet and vest boxes with a confidence score for each detection.'),
    ('03', 'Associate equipment', 'Match helmets to plausible head regions and vests to torsos. Reject uncertain or competing ownership.'),
    ('04', 'Explain the evidence', 'Apply confidence, scale, border and overlap checks. Missing detections default to human review.'),
    ('05', 'Track in video', 'Optional geometric session IDs connect nearby observations. Repeated evidence confirms green status.'),
    ('06', 'Review and export', 'Draw annotations on a copy of the frame. Save JSON evidence, source hashes and measured video timing.'),
)


def pipeline_html():
    return '<div class="pipeline">' + ''.join(
        f'<div class="pipeline-card"><b>{number} / PIPELINE</b><h3>{title}</h3><p>{description}</p></div>'
        for number, title, description in PIPELINE) + '</div>'


def status_html(status):
    style = {'review required': 'review', 'potential violation': 'violation'}.get(status, '')
    return f'<span class="status-pill {style}">{escape(status)}</span>'


def empty_stage(is_video=False):
    import streamlit as st

    title = 'Turn footage into<br>visible evidence.' if is_video else 'Every worker.<br>A clearer picture.'
    description = ('Preview footage and live AI evidence in this browser, or analyze every frame for a downloadable report.'
                   if is_video else 'Add a worksite image to inspect people, helmets and safety vests. Review the evidence behind each worker’s result.')
    st.markdown(f'''<div class="empty-stage"><div class="eyebrow">READY WHEN YOU ARE</div>
        <h2>{title}</h2><p>{description}</p>
        <div class="scan-graphic" aria-hidden="true"><div class="person"></div><div class="head"></div>
        <div class="vest"></div><div class="line"></div><div class="label">PERSON · HELMET · VEST</div></div></div>''', unsafe_allow_html=True)


def worker_evidence(assessment):
    import streamlit as st
    from ppe_compliance.rendering import worker_scores

    workers = assessment['workers']
    if not workers:
        st.info('No workers detected. This does not establish an empty or safe scene.')
        return
    status = st.selectbox('Filter worker evidence', ['All workers', 'review required', 'PPE detected', 'potential violation'])
    shown = [w for w in workers if status == 'All workers' or w['status'] == status]
    if not shown:
        st.info('No workers have this status in the current image.')
    for worker in shown:
        with st.container(border=True):
            left, right = st.columns([1, 2])
            left.markdown(f"**{worker['worker_id']}**")
            right.markdown(status_html(worker['status']), unsafe_allow_html=True)
            person, helmet, vest = worker_scores(worker)
            st.caption(f'Person {person} · Helmet {helmet} · Vest {vest}')
            st.write(f"Helmet: **{worker['helmet']['state']}** · Vest: **{worker['vest']['state']}**")
            for reason in worker['reasons']:
                st.caption(reason)


def project_guide(root):
    import streamlit as st

    st.caption('FROM PIXELS TO WORKER EVIDENCE')
    st.markdown(pipeline_html(), unsafe_allow_html=True)
    st.subheader('Three outcomes, with reasons')
    st.dataframe([
        {'Outcome': 'PPE detected', 'Meaning': 'Both required items plausibly match, with no quality or visibility warning.'},
        {'Outcome': 'review required', 'Meaning': 'Missing, small, ambiguous, low-confidence or occluded evidence needs a reviewer.'},
        {'Outcome': 'potential violation', 'Meaning': 'An item is unconfirmed in a human-reviewed visible region. Available through CLI visibility input.'},
    ], hide_index=True, width='stretch')
    st.info('The detector finds objects. Geometry assigns plausible ownership. Rules explain uncertainty. None of these proves wearing state or overall workplace safety.')
    with st.expander('Know the code before presenting', expanded=True):
        st.markdown('''1. `detector.py → FrameDetector.detect`: model prediction, valid boxes and scores.
2. `association.py → associate_ppe`: head/torso geometry and one-worker ownership.
3. `compliance.py → assess_detections`: conservative statuses and explanations.
4. `tracking.py → WorkerTracker.update`: session IDs, expiry and confirmation.
5. `video_pipeline.py → process_video`: every-frame exports and measured throughput.
6. `live_pipeline.py` / `recorded_preview.py`: separate source playback and sampled AI.
7. `interface.py` / `streamlit_app.py`: upload validation, session state and downloads.''')
    st.subheader('Presentation reading pack')
    for name, label in [('pipeline.md', 'Pipeline & architecture'), ('code-walkthrough.md', 'Code walkthrough & viva questions'),
                        ('presentation-guide.md', 'Demo script & troubleshooting'), ('final-audit.md', 'Final audit & verification')]:
        path = root/'docs'/name
        if path.is_file():
            st.download_button(label, path.read_bytes(), file_name=name, mime='text/markdown', key='guide-'+name)


def validation_view(root):
    import streamlit as st

    st.caption('RECORDED DEVELOPMENT EVIDENCE')
    path = root/'docs/experiments/baseline-001/validation-metrics.json'
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        aggregate = data['aggregate']
        cols = st.columns(4)
        for col, label, key in zip(cols, ['Precision', 'Recall', 'mAP @ 0.5', 'mAP @ 0.5:0.95'],
                                   ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']):
            col.metric(label, f'{aggregate[key]*100:.2f}%')
        st.caption('Validation split · 200 images · YOLO11 nano · 30 epochs · Colab Tesla T4. Metrics are read from the saved baseline artifact.')
        st.subheader('Where the model needs attention')
        st.dataframe([{'Class': row['class'].title(), 'Precision': f"{row['precision']:.2%}",
                       'Recall': f"{row['recall']:.2%}", 'mAP @ 0.5': f"{row['mAP50']:.2%}"}
                      for row in data['per_class']], hide_index=True, width='stretch')
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.warning(f'Baseline metrics are unavailable: {error}')
    st.warning('Independent final evaluation is pending. Validation mAP is not worker-level compliance accuracy; recurring scenes limit independence.')
    st.subheader('What the checks establish')
    st.write('Automated checks exercise rules, input errors, tracking, video timing, export failures and interface behavior. Real-model smoke checks verify local integration. Independent human-labeled data is required to measure final accuracy.')
    st.code('python -m unittest discover -s tests -v\npython scripts/verify_presentation.py', language='powershell')
    st.caption('Processing FPS, moving-preview FPS and output playback FPS measure different things. Report hardware, input size and run duration with performance claims.')
