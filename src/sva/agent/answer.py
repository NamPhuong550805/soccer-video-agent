"""Answer Agent: cau tra loi tieng Anh theo khuon cho tung loai cau hoi, kem do tin, moc thoi gian that va frame bang chung.
Do tin = xac suat cao nhat cua phan phoi cuoi; duoi nguong (chon tren dev bang risk-coverage) thi tu choi co ly do."""
from pathlib import Path

import cv2

from sva.perception.video import sample_frames, split_segment


def clock(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def span(path):
    file, a, b = split_segment(path)
    return f"{clock(a)}-{clock(b)}" if a is not None else "the whole video"


def save_frame(path, out_dir, name):
    """Frame giua doan -> PNG; tra ve duong dan file."""
    frames, _ = sample_frames(path, 1)
    out = Path(out_dir) / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    return str(out)


def compose(sample, label, prob, dist_labels, abstain_below=0.0):
    """(cau tra loi, trang thai). dist_labels: [(nhan, xac suat)] giam dan."""
    t, where = sample.task_id, span(sample.materials[0])
    if prob < abstain_below:
        (a, pa), (b, pb) = dist_labels[0], dist_labels[1] if len(dist_labels) > 1 else ("-", 0.0)
        return (f"Not sure. Most likely: {a} ({pa:.0%}), then {b} ({pb:.0%}). Please check {where}.", "abstained")
    conf = f"(confidence {prob:.2f})"
    if t == "q9":
        return f"The replay at {where} shows the play at {label} {conf}.", "answered"
    if t == "q8":
        return f"At {where} the camera changes {label[0].lower() + label[1:].rstrip('.')} {conf}.", "answered"
    if t == "q10":
        return f"At {where}: {label} {conf}.", "answered"
    if t == "q11":
        return f"Commentary for {where}: \"{label}\" {conf}", "answered"
    if t == "q13":
        return f"The player wears a {label} jersey {conf}.", "answered"
    return f"{label} {conf}.", "answered"
