"""Draw the same worker evidence represented in the compliance JSON."""


def draw_assessment(frame, assessment, detections):
    import cv2

    image = frame.copy()
    height, width = image.shape[:2]
    scale = max(.4, min(1.0, max(width, height)/1500))
    line_height = max(20, round(35*scale))
    thickness = 2 if scale >= .8 else 1
    colors = {"PPE detected": (50, 150, 20), "potential violation": (20, 20, 210),
              "review required": (0, 130, 210)}
    occupied = []

    def collides(box):
        return any(box[0] < other[2]+3 and box[2]+3 > other[0] and
                   box[1] < other[3]+3 and box[3]+3 > other[1] for other in occupied)

    def worker_label(lines, left, top, color):
        text_height = cv2.getTextSize(lines[0], cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][1]
        block_width = min(width, max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
                                     for line in lines)+8)
        block_height = len(lines)*line_height
        x = max(0, min(left, width-block_width))
        preferred = max(0, min(top, height-block_height-line_height-5))
        y = preferred
        for offset in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6):
            candidate = max(0, min(preferred+offset*(block_height+5), height-block_height-line_height-5))
            box = (x, candidate, x+block_width, candidate+block_height)
            if not collides(box):
                y = candidate
                break
        occupied.append((x,y,x+block_width,y+block_height))
        if abs(y-top) > line_height:
            cv2.line(image, (left, top), (x, y+block_height//2), color, 1)
        cv2.rectangle(image, (x,y), (min(width-1,x+block_width),min(height-1,y+block_height)), color, -1)
        for index, line in enumerate(lines):
            cv2.putText(image, line, (x+3,y+index*line_height+text_height+5), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255,255,255), thickness, cv2.LINE_AA)

    def label(text, x, y, color):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
        x = max(0, min(x, width-text_size[0]-8))
        y = max(text_size[1]+5, min(y, height-4))
        cv2.rectangle(image, (x, y-text_size[1]-5), (min(width-1, x+text_size[0]+7), y+3), color, -1)
        cv2.putText(image, text, (x+3, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness, cv2.LINE_AA)

    for detection in detections:
        kind = detection["class_name"].casefold()
        if kind in {"helmet", "vest"}:
            left, top, right, bottom = [round(v) for v in detection["bbox_xyxy"]]
            cv2.rectangle(image, (left, top), (right, bottom), (190, 90, 50), 1)
    for worker in assessment["workers"]:
        left, top, right, bottom = [round(v) for v in worker["bbox_xyxy"]]
        color = colors[worker["status"]]
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        compact = right-left < 100*scale or bottom-top < 140*scale
        if compact:
            short = {"PPE detected":"PPE", "review required":"review", "potential violation":"check"}
            lines = [f"{worker['worker_id']}: {short[worker['status']]}"]
        else:
            lines = [f"{worker['worker_id']}: {worker['status']}",
                     f"helmet: {worker['helmet']['state']} | vest: {worker['vest']['state']}"]
        worker_label(lines, left, top, color)
    label(f"Detected workers: {len(assessment['workers'])} | Visible PPE only; human review may be needed",
          5, height-10, (40,40,40))
    return image
