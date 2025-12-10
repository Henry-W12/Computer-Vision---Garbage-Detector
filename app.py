import streamlit as st
import cv2
import numpy as np
import joblib
import json
import time
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from io import BytesIO

# --------------------------- Helper functions ---------------------------

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
    if scaler is None:
        return None
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

# --------------------------- Bounding Box Mode A ---------------------------

def _bbox_mode_A(frame, min_area):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    th = cv2.adaptiveThreshold(blur, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return []
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    if w*h < min_area:
        return []
    return [(x,y,w,h,w*h)]

def get_candidate_bboxes(frame, min_area=1500, max_bbox=6):
    boxes = _bbox_mode_A(frame, min_area)
    return boxes[:max_bbox]

# --------------------------- Streamlit App ---------------------------

st.set_page_config(page_title="Garbage Detector", layout="wide")
st.title("📦 Garbage Detector Using XGBoost")

# --------------------------- Sidebar ---------------------------

with st.sidebar:
    st.header("Model & Input")
    stage1_path = st.text_input("Stage 1 pipeline (.joblib)", value="stage1_xgb_augmented.joblib")
    stage2_path = st.text_input("Stage 2 pipeline (.joblib)", value="stage2_pipeline.joblib")
    threshold_path = st.text_input("Threshold JSON", value="stage1_threshold.json")
    mode = st.selectbox("Mode", ["Image", "Webcam (one-shot)", "Local Camera (live loop)"])

    with st.expander("Advanced settings"):
        min_area = st.number_input("Min proposal area", value=2500, step=500)
        max_boxes = st.number_input("Max proposals per frame", value=6, min_value=1, max_value=20)
        resize = st.number_input("Resize for feature extraction", value=128)

    run_button = st.button("Load models & Start")

load_status = st.empty()
canvas = st.empty()
info = st.empty()

if 'models_loaded' not in st.session_state:
    st.session_state['models_loaded'] = False

# --------------------------- Load Models ---------------------------

if run_button:
    try:
        with load_status.container():
            st.info("Loading models...")
        p1 = joblib.load(stage1_path)
        p2 = joblib.load(stage2_path)
        cfg = json.load(open(threshold_path))

        scaler1, pca1, model1 = p1['scaler'], p1.get('pca', None), p1['model']
        le1 = p1.get('label_encoder', None)
        scaler2, pca2, model2 = p2['scaler'], p2.get('pca', None), p2['model']

        POS = cfg.get('positive_class','Non-Organik')
        BASE_THR = float(cfg.get('threshold',0.5))
        class_names1 = list(le1.inverse_transform(np.arange(len(le1.classes_)))) if le1 is not None else (list(getattr(model1,'classes_',[])))
        class_names2 = list(getattr(model2,'classes_',[]))

        expected1 = get_expected_dim(scaler1, pca1)
        expected2 = get_expected_dim(scaler2, pca2)

        if expected1 is None or expected2 is None:
            dummy = np.zeros((resize,resize,3), dtype=np.uint8)
            feat_len = extract_features_from_img(dummy, resize=(resize,resize)).ravel().size
            if expected1 is None:
                expected1 = feat_len
            if expected2 is None:
                expected2 = feat_len

        model_dict = {
            'scaler1':scaler1,'pca1':pca1,'model1':model1,'le1':le1,
            'scaler2':scaler2,'pca2':pca2,'model2':model2,
            'POS':POS,'BASE_THR':BASE_THR,'class_names1':class_names1,'class_names2':class_names2,
            'expected1':expected1,'expected2':expected2,
        }

        st.session_state['models'] = model_dict
        st.session_state['models_loaded'] = True
        load_status.success("Models loaded successfully.")
        st.write("Loaded model keys:", list(model_dict.keys()))
    except Exception as e:
        load_status.error(f"Failed to load models: {e}")
        st.session_state['models_loaded'] = False

# --------------------------- Inference ---------------------------

def run_inference_on_frame(frame):
    if not st.session_state.get('models_loaded', False):
        return frame
    m = st.session_state.get('models', {})
    out = frame.copy()
    boxes = get_candidate_bboxes(frame, min_area=min_area, max_bbox=max_boxes)

    for (x,y,w,h,area) in boxes:
        x = max(0,x); y = max(0,y); w = max(1,w); h = max(1,h)
        w = min(w, frame.shape[1]-x); h = min(h, frame.shape[0]-y)

        roi = frame[y:y+h, x:x+w].copy()
        if roi.size == 0:
            label, conf = 'Unknown', 0.0
        else:
            feat = extract_features_from_img(roi, resize=(resize,resize)).astype(np.float32)
            fv1 = align_feature(feat, m.get('expected1'))
            prob_pos = 0.0
            try:
                X1 = m['scaler1'].transform(fv1.reshape(1,-1))
                if m.get('pca1'): X1 = m['pca1'].transform(X1)
                probs1 = m['model1'].predict_proba(X1)[0]
                if m['class_names1'] and m['POS'] in m['class_names1']:
                    prob_pos = float(probs1[m['class_names1'].index(m['POS'])])
                else:
                    prob_pos = float(np.max(probs1))
            except Exception:
                prob_pos = 0.0

            if prob_pos >= float(m.get('BASE_THR',0.5)):
                fv2 = align_feature(feat, m.get('expected2'))
                try:
                    X2 = m['scaler2'].transform(fv2.reshape(1,-1))
                    if m.get('pca2'): X2 = m['pca2'].transform(X2)
                    probs2 = m['model2'].predict_proba(X2)[0]
                    best_idx = int(np.argmax(probs2))
                    label = m['class_names2'][best_idx] if m.get('class_names2') else "Unknown"
                    conf = float(probs2[best_idx])
                except Exception:
                    label, conf = "Unknown", 0.0
            else:
                label, conf = 'Organic', float(1.0-prob_pos)

        # Draw transparent bbox + text
        overlay = out.copy()
        color = (0,0,255) if label != 'Organic' else (0,200,0)
        alpha = 0.3
        cv2.rectangle(overlay, (x,y), (x+w, y+h), color, -1)
        cv2.addWeighted(overlay, alpha, out, 1-alpha, 0, out)

        # Text with outline
        txt = f"{label} {conf:.2f}"
        (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x, y-22), (x+tw+6, y), (0,0,0), -1)
        cv2.putText(out, txt, (x+3, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    return out

# --------------------------- Modes ---------------------------

def display_frame(frame):
    MAX_W = 800
    h, w = frame.shape[:2]
    scale = min(1.0, float(MAX_W)/w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

if mode == 'Image':
    uploaded = st.file_uploader("Upload an image", type=['jpg','jpeg','png'])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        out = run_inference_on_frame(frame)
        canvas.image(display_frame(out), channels='RGB')
        info.info("Inference complete")
        # Download button
        _, im_buf_arr = cv2.imencode(".png", out)
        st.download_button("Download Result", im_buf_arr.tobytes(), "result.png", "image/png")

elif mode == 'Webcam (one-shot)':
    cam_file = st.camera_input("Take a photo (webcam)")
    if cam_file is not None:
        file_bytes = np.asarray(bytearray(cam_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        out = run_inference_on_frame(frame)
        canvas.image(display_frame(out), channels='RGB')
        # Download
        _, im_buf_arr = cv2.imencode(".png", out)
        st.download_button("Download Result", im_buf_arr.tobytes(), "result.png", "image/png")

else:  # Live camera
    live_run = st.checkbox("Live Camera")
    cam_placeholder = st.empty()
    if live_run:
        src = 0
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            st.error("Cannot open local camera. Try Webcam (one-shot).")
        else:
            try:
                while live_run:
                    ret, frame = cap.read()
                    if not ret: break
                    out = run_inference_on_frame(frame)
                    cam_placeholder.image(display_frame(out), channels='RGB')
                    time.sleep(0.02)
                    live_run = st.checkbox("Live Camera", value=True)
            finally:
                cap.release()

st.markdown("---")
st.caption("📌 This demo uses XGBoost pipelines for garbage detection. Use Advanced Settings for tweaking parameters.")
