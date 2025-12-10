import cv2, joblib, json, time, argparse, os, numpy as np
from collections import deque
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def color_hist_hsv(img, bins=(8,8,8)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,180,0,256,0,256])
    if hist.sum()!=0: hist = hist / hist.sum()
    return hist.flatten()

def color_moments(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float32')
    feats=[]
    for i in range(3):
        ch = hsv[:,:,i].ravel()
        feats.append(np.mean(ch)); feats.append(np.std(ch))
        feats.append(((ch-ch.mean())**3).mean()/(ch.std()**3) if ch.std()>0 else 0.0)
    return np.array(feats, dtype=np.float32)

def lbp_feature(img_gray, P=8, R=1):
    lbp = local_binary_pattern(img_gray, P, R, method='uniform')
    n_bins = int(lbp.max()+1)
    hist,_ = np.histogram(lbp.ravel(), bins=n_bins, range=(0,n_bins), density=True)
    return hist.astype(np.float32)

def glcm_features(img_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32):
    img = img_gray.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx>mn:
        img_q = ((img-mn)/(mx-mn)*(levels-1)).round().astype(np.uint8)
    else:
        img_q = np.zeros_like(img, dtype=np.uint8)
    glcm = graycomatrix(img_q, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    props = ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']
    feats=[]
    for p in props:
        feats.extend(graycoprops(glcm,p).flatten())
    return np.array(feats, dtype=np.float32)

def hu_moments(img_gray):
    m = cv2.moments(img_gray)
    hu = cv2.HuMoments(m).flatten()
    return np.array([-np.sign(v)*np.log10(abs(v)+1e-10) for v in hu], dtype=np.float32)

def extract_features_from_img(img, resize=(128,128)):
    im = cv2.resize(img, resize)
    ig = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    parts=[]
    parts.append(color_hist_hsv(im))
    parts.append(color_moments(im))
    parts.append(lbp_feature(ig))
    parts.append(glcm_features(ig))
    parts.append(hu_moments(ig))
    return np.concatenate(parts)

def get_expected_dim(scaler, pca):
    if hasattr(scaler,'mean_') and getattr(scaler,'mean_',None) is not None:
        return int(np.asarray(scaler.mean_).shape[0])
    if hasattr(scaler,'n_features_in_') and getattr(scaler,'n_features_in_',None) is not None:
        return int(scaler.n_features_in_)
    if pca is not None and hasattr(pca,'n_features_in_') and getattr(pca,'n_features_in_',None) is not None:
        return int(pca.n_features_in_)
    return None

def align_feature(fv, expected):
    fv = np.asarray(fv).ravel()
    if expected is None: return fv
    cur = fv.size
    if cur == expected: return fv
    if cur < expected:
        pad = np.zeros((expected-cur,), dtype=fv.dtype)
        return np.concatenate([fv,pad])
    return fv[:expected]

def get_candidate_bboxes(frame, min_area=2000, max_bbox=6):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h >= min_area:
            boxes.append((x,y,w,h,w*h))
    boxes = sorted(boxes, key=lambda t: t[4], reverse=True)[:max_bbox]
    return boxes

def create_tracker():
    def try_get(fn_name):
        try:
            parts = fn_name.split('.')
            attr = cv2
            for p in parts:
                attr = getattr(attr, p)
            return attr
        except Exception:
            return None

    factories = [
        "TrackerCSRT_create",
        "TrackerKCF_create",
        "TrackerMOSSE_create",
        "TrackerMIL_create",
        "TrackerBoosting_create",
        "legacy.TrackerCSRT_create",
        "legacy.TrackerKCF_create",
        "legacy.TrackerMOSSE_create",
        "legacy.TrackerMIL_create",
        "legacy.TrackerBoosting_create"
    ]

    for name in factories:
        factory = try_get(name)
        if factory is not None:
            try:
                return factory()
            except Exception:
                pass

    for name in ["TrackerCSRT_create", "TrackerKCF_create", "TrackerMOSSE_create"]:
        try:
            factory = getattr(cv2, "legacy_"+name)
            return factory()
        except Exception:
            pass

    class LKTracker:
        def __init__(self):
            self.prev_gray = None
            self.points = None
            self.bbox = None

        def init(self, frame, bbox):
            x,y,w,h = [int(v) for v in bbox]
            self.bbox = (x,y,w,h)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_gray = gray
            mask = np.zeros_like(gray)
            mask[y:y+h, x:x+w] = 255
            p = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=3, mask=mask)
            if p is not None:
                self.points = p
            else:
                self.points = None

        def update(self, frame):
            if self.prev_gray is None or self.bbox is None:
                return False, (0,0,0,0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.points is None or len(self.points) == 0:
                x,y,w,h = self.bbox
                mask = np.zeros_like(gray)
                mask[y:y+h, x:x+w] = 255
                p = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=3, mask=mask)
                self.points = p
                self.prev_gray = gray
                if p is None:
                    return False, (0,0,0,0)
            nextPts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.points, None, winSize=(15,15), maxLevel=2,
                                                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            if nextPts is None:
                return False, (0,0,0,0)
            good_new = nextPts[status==1]
            good_old = self.points[status==1]
            if len(good_new) < 3:
                self.prev_gray = gray
                self.points = good_new.reshape(-1,1,2) if len(good_new)>0 else None
                return False, (0,0,0,0)
            disp = (good_new - good_old).median(axis=0)
            dx, dy = float(disp[0]), float(disp[1])
            x,y,w,h = self.bbox
            x_new = int(round(x + dx)); y_new = int(round(y + dy))
            self.bbox = (x_new, y_new, w, h)
            self.points = good_new.reshape(-1,1,2)
            self.prev_gray = gray
            return True, (x_new, y_new, w, h)

    return LKTracker()

def iou_box(boxA, boxB):
    xA,yA,wA,hA = boxA; xB,yB,wB,hB = boxB
    xa1, ya1, xa2, ya2 = xA, yA, xA+wA, yA+hA
    xb1, yb1, xb2, yb2 = xB, yB, xB+wB, yB+hB
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    iw = max(0, xi2-xi1); ih = max(0, yi2-yi1)
    inter = iw*ih
    areaA = (xa2-xa1)*(ya2-ya1)
    areaB = (xb2-xb1)*(yb2-yb1)
    union = areaA + areaB - inter
    return inter/union if union>0 else 0.0

class LockTrack:
    def __init__(self, tid, bbox, init_frame, ema_alpha=0.6, confirm_frames=3):
        self.id = tid
        self.bbox = bbox
        self.tracker = create_tracker()
        self.tracker.init(frame_global, tuple(bbox))
        self.ema_alpha = ema_alpha
        self.ema_box = np.array(bbox, dtype=float)
        self.confirm_count = 1
        self.confirm_frames = confirm_frames
        self.confirmed = False
        self.label = None
        self.conf = 0.0
        self.missed = 0
        self.last_update_frame = init_frame

    def update_from_tracker(self, frame):
        ok, bb = self.tracker.update(frame)
        if not ok:
            self.missed += 1
            return False
        x,y,w,h = [int(v) for v in bb]
        self.ema_box = self.ema_alpha * np.array([x,y,w,h], dtype=float) + (1-self.ema_alpha) * self.ema_box
        self.bbox = [int(v) for v in self.ema_box]
        self.missed = 0
        return True

    def touch_confirm(self):
        self.confirm_count += 1
        if not self.confirmed and self.confirm_count >= self.confirm_frames:
            self.confirmed = True

def run_realtime(args):
    global frame_global
    p1 = joblib.load(args.stage1)
    p2 = joblib.load(args.stage2)
    cfg = json.load(open(args.threshold))
    scaler1, pca1, model1 = p1['scaler'], p1.get('pca', None), p1['model']
    le1 = p1.get('label_encoder', None)
    scaler2, pca2, model2 = p2['scaler'], p2.get('pca', None), p2['model']

    POS = cfg.get('positive_class', 'Non-Organik'); BASE_THR = float(cfg.get('threshold',0.5))
    class_names1 = list(le1.inverse_transform(np.arange(len(le1.classes_)))) if le1 is not None else list(model1.classes_)
    class_names2 = list(model2.classes_)
    expected1 = get_expected_dim(scaler1, pca1)
    expected2 = get_expected_dim(scaler2, pca2)

    print("Loaded pipelines. expected dims (s1,s2):", expected1, expected2)
    print("Stage1 classes:", class_names1, "POS:", POS, "BASE_THR:", BASE_THR)
    print("Stage2 classes:", class_names2)
    print("Starting camera...")

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera. Check permissions/index/OpenCV build.")

    tracks = dict()
    next_tid = 1

    frame_idx = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame empty. Exiting.")
            break
        frame_global = frame.copy()
        frame_idx += 1

        to_delete = []
        for tid, tr in tracks.items():
            ok = tr.update_from_tracker(frame)
            if not ok:
                tr.missed += 1
            if tr.missed > args.max_missed:
                to_delete.append(tid)
        for tid in to_delete:
            del tracks[tid]

        if frame_idx % args.proposal_every == 0:
            boxes = get_candidate_bboxes(frame, min_area=args.min_area, max_bbox=args.max_boxes)
            matched_prop = set()
            for (x,y,w,h,area) in boxes:
                best_tid = None; best_iou = 0.0
                for tid, tr in tracks.items():
                    bx,by,bw,bh = tr.bbox
                    val = iou_box((x,y,w,h),(bx,by,bw,bh))
                    if val > best_iou:
                        best_iou = val; best_tid = tid
                if best_iou >= args.iou_match and best_tid is not None:
                    tracks[best_tid].touch_confirm()
                    matched_prop.add(best_tid)
                else:
                    tid = next_tid; next_tid += 1
                    bbox = [x,y,w,h]
                    tr = LockTrack(tid, bbox, frame_idx, ema_alpha=args.ema_alpha, confirm_frames=args.confirm_frames)
                    tracks[tid] = tr
                    matched_prop.add(tid)

            for tid, tr in list(tracks.items()):
                if tr.confirmed and tr.label is None:
                    x,y,w,h = tr.bbox
                    roi = frame[y:y+h, x:x+w].copy()
                    if roi.size == 0:
                        tr.label = "Unknown"; tr.conf = 0.0
                    else:
                        feat = extract_features_from_img(roi, resize=(args.resize,args.resize)).astype(np.float32)
                        fv1 = align_feature(feat, expected1)
                        X1 = scaler1.transform(fv1.reshape(1,-1))
                        if pca1 is not None: X1 = pca1.transform(X1)
                        probs1 = model1.predict_proba(X1)[0]
                        prob_pos = float(probs1[class_names1.index(POS)]) if POS in class_names1 else max(probs1)
                        if prob_pos >= BASE_THR:
                            fv2 = align_feature(feat, expected2)
                            X2 = scaler2.transform(fv2.reshape(1,-1))
                            if pca2 is not None: X2 = pca2.transform(X2)
                            probs2 = model2.predict_proba(X2)[0]
                            best_idx = int(np.argmax(probs2))
                            tr.label = class_names2[best_idx]
                            tr.conf = float(probs2[best_idx])
                        else:
                            tr.label = "Organic"
                            tr.conf = 1-prob_pos

        for tid, tr in tracks.items():
            x,y,w,h = tr.bbox
            if tr.confirmed:
                color = (0,0,255) if tr.label != "Organic" else (0,200,0)
                txt = f"{tr.label} {tr.conf:.2f} id:{tid}"
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y-22), (x+tw+6, y), color, -1)
                cv2.putText(frame, txt, (x+3, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            else:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (200,200,200), 1)
                cv2.putText(frame, f"pending id:{tid} ({tr.confirm_count}/{tr.confirm_frames})", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        elapsed = time.time() - start
        fps = frame_idx / elapsed if elapsed>0 else 0.0
        cv2.putText(frame, f"FPS:{fps:.1f} tracks:{len(tracks)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.imshow("Realtime Lock-on", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"snap_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print("Saved", fname)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--stage1', default='stage1_xgb_calib.joblib')
    p.add_argument('--stage2', default='stage2_pipeline.joblib')
    p.add_argument('--threshold', default='stage1_threshold.json')
    p.add_argument('--source', default='0')
    p.add_argument('--proposal_every', type=int, default=3, help='how often (frames) to run proposals')
    p.add_argument('--confirm_frames', type=int, default=3, help='times a proposal must be matched to confirm')
    p.add_argument('--max_missed', type=int, default=8, help='frames allowed missed before removing track')
    p.add_argument('--min_area', type=int, default=2500)
    p.add_argument('--max_boxes', type=int, default=6)
    p.add_argument('--resize', type=int, default=128)
    p.add_argument('--ema_alpha', type=float, default=0.6)
    p.add_argument('--iou_match', type=float, default=0.3)
    args = p.parse_args()
    run_realtime(args)
