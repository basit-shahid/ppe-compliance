"""Small worker markers; full evidence belongs in the accompanying report."""


def confidence_label(value):
    """Model confidence, never an invented compliance/safety probability."""
    import math

    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or not 0 <= value <= 1):
        return '--'
    return f'{value:.0%}'


def worker_scores(worker):
    return (confidence_label(worker.get('person_confidence')),
            confidence_label(worker['helmet'].get('confidence')),
            confidence_label(worker['vest'].get('confidence')))


def worker_badge(worker, show_scores=True):
    short = {"PPE detected": "PPE", "potential violation": "Check", "review required": "Review"}
    text = f"{worker['worker_id']}  {short[worker['status']]}"
    if show_scores:
        person, helmet, vest = worker_scores(worker)
        text += f' | P{person} H{helmet} V{vest}'
    return text


def resize_frame(frame, size):
    """Area-reduce large, evenly divisible frames before the final pane resize.

    Integer area reduction avoids the expensive arbitrary-ratio 4K filter while
    retaining area filtering. Callers choose whether this is display or analysis.
    """
    import cv2

    height, width = frame.shape[:2]
    factor = 1
    while (width % (factor*2) == 0 and height % (factor*2) == 0 and
           width//(factor*2) >= size[0] and height//(factor*2) >= size[1]):
        factor *= 2
    if factor > 1:
        frame = cv2.resize(frame, (width//factor, height//factor), interpolation=cv2.INTER_AREA)
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def draw_assessment(frame, assessment, detections, *, detail=False, show_scores=True, fit=None):
    """Scale report coordinates to the rendered frame without mutating evidence."""
    import cv2

    source_height, source_width = frame.shape[:2]
    ratio = min(1., fit[0]/source_width, fit[1]/source_height) if fit else 1.
    image = (resize_frame(frame, (max(1, round(source_width*ratio)), max(1, round(source_height*ratio))))
             if ratio < 1 else frame.copy())
    height, width = image.shape[:2]
    coordinate_width = assessment.get('image_width', source_width)
    coordinate_height = assessment.get('image_height', source_height)

    def display_box(box):
        return [round(value*(width/coordinate_width if i % 2 == 0 else height/coordinate_height))
                for i, value in enumerate(box)]

    scale = max(.4, min(1.15, max(width, height) / 1600))
    stroke = max(1, round(max(width, height) / 1000))
    colors = {"PPE detected": (102, 206, 65), "potential violation": (85, 85, 239),
              "review required": (60, 191, 245)}
    occupied = []
    if detail:
        for detection in detections:
            if detection["class_name"].casefold() in {"helmet", "vest"}:
                left, top, right, bottom = display_box(detection["bbox_xyxy"])
                cv2.rectangle(image, (left, top), (right, bottom), (176, 154, 95), 1, cv2.LINE_AA)
                label = detection['class_name'].capitalize()
                if show_scores:
                    label += ' ' + confidence_label(detection.get('confidence'))
                cv2.putText(image, label, (max(0, left), max(12, top-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, scale*.8, (240, 244, 248), 1, cv2.LINE_AA)
    for worker in assessment["workers"]:
        left, top, right, bottom = display_box(worker["bbox_xyxy"])
        left, right = max(0, min(width-1, left)), max(0, min(width-1, right))
        top, bottom = max(0, min(height-1, top)), max(0, min(height-1, bottom))
        if right <= left or bottom <= top:
            continue
        color = colors[worker["status"]]
        corner = max(2, min(18, (right-left)//4, (bottom-top)//4))
        for x, dx in ((left, 1), (right, -1)):
            for y, dy in ((top, 1), (bottom, -1)):
                cv2.line(image, (x, y), (x+dx*corner, y), color, stroke, cv2.LINE_AA)
                cv2.line(image, (x, y), (x, y+dy*corner), color, stroke, cv2.LINE_AA)
        text = worker_badge(worker, show_scores)
        if detail:
            text += f" | H: {worker['helmet']['state']} V: {worker['vest']['state']}"
        while text and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width-10:
            text = text[:-1]
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        bw, bh = min(width, tw+10), min(height, th+baseline+8)
        x = max(0, min(left, width-bw))
        chosen = None
        for proposed in (top-bh-3, top+3, bottom+3, bottom-bh-3):
            y = max(0, min(proposed, height-bh))
            box = (x, y, x+bw, y+bh)
            if not any(box[0] < b[2]+2 and box[2]+2 > b[0] and
                       box[1] < b[3]+2 and box[3]+2 > b[1] for b in occupied):
                chosen = box
                break
        # Dense crowds retain corners instead of stacking unreadable labels.
        if chosen is None:
            continue
        occupied.append(chosen)
        x, y = chosen[:2]
        cv2.rectangle(image, (x, y), (x+bw-1, y+bh-1), (28, 33, 40), -1)
        cv2.rectangle(image, (x, y), (x+2, y+bh-1), color, -1)
        cv2.putText(image, text, (x+6, y+th+3), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (240, 244, 248), 1, cv2.LINE_AA)
    return image
