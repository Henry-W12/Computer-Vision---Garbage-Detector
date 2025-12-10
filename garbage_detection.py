#imports
import os, sys, argparse, random, time, json, shutil
from pathlib import Path
from multiprocessing import cpu_count
from collections import Counter
import cv2
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from joblib import Parallel, delayed

#augmentations
def augment_image(img, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    out = img.copy()
    h, w = out.shape[:2]

    r = random.random()
    if r < 0.33:
        out = cv2.flip(out, 1)
    elif r < 0.66:
        out = cv2.flip(out, 0)

    angle = random.uniform(-25, 25)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    
    alpha = random.uniform(0.9, 1.2)
    beta = random.uniform(-20, 20)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if random.random() < 0.3:
        k = random.choice([3,5])
        out = cv2.GaussianBlur(out, (k,k), 0)

    if random.random() < 0.5:
        scale = random.uniform(0.8, 1.0)
        nh, nw = int(h*scale), int(w*scale)
        y0 = random.randint(0, h-nh) if h>nh else 0
        x0 = random.randint(0, w-nw) if w>nw else 0
        crop = out[y0:y0+nh, x0:x0+nw]
        out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)

    if random.random() < 0.5:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[...,1] = np.clip(hsv[...,1] + random.randint(-15,15), 0, 255)
        hsv[...,2] = np.clip(hsv[...,2] + random.randint(-15,15), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return out

def color_hist_hsv(img, bins=(8,8,8)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, bins, [0,180,0,256,0,256])
    if hist.sum() != 0:
        hist = hist / hist.sum()
    return hist.flatten()

def color_moments(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float32')
    feats = []
    for i in range(3):
        ch = hsv[:,:,i].ravel()
        feats.append(np.mean(ch))
        feats.append(np.std(ch))
        feats.append(((ch - ch.mean())**3).mean() / (ch.std()**3) if ch.std() > 0 else 0.0)
    return np.array(feats, dtype=np.float32)

def lbp_feature(img_gray, P=8, R=1):
    lbp = local_binary_pattern(img_gray, P, R, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)

def glcm_features(img_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32):
    img = img_gray.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx > mn:
        img_q = ((img - mn) / (mx - mn) * (levels - 1)).round().astype(np.uint8)
    else:
        img_q = np.zeros_like(img, dtype=np.uint8)
    glcm = graycomatrix(img_q, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    props = ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']
    feats = []
    for p in props:
        feats.extend(graycoprops(glcm, p).flatten())
    return np.array(feats, dtype=np.float32)

def hu_moments(img_gray):
    moments = cv2.moments(img_gray)
    hu = cv2.HuMoments(moments).flatten()
    for i in range(len(hu)):
        hu[i] = -np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-10)
    return hu.astype(np.float32)

def extract_features(img_path_or_img, resize=(128,128), use_hog=False):
    if isinstance(img_path_or_img, str):
        img = cv2.imread(img_path_or_img)
        if img is None:
            raise ValueError(f"Cannot read {img_path_or_img}")
    else:
        img = img_path_or_img
    img = cv2.resize(img, resize)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feats = []
    feats.append(color_hist_hsv(img, bins=(8,8,8)))
    feats.append(color_moments(img))
    feats.append(lbp_feature(img_gray, P=8, R=1))
    feats.append(glcm_features(img_gray, levels=32))
    feats.append(hu_moments(img_gray))
    if use_hog:
        feats.append(hog(img_gray, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2), feature_vector=True))
    return np.concatenate(feats)

def _proc_and_cache(tpl, cache_dir, resize, use_hog):
    p, lbl = tpl
    base = os.path.splitext(os.path.basename(p))[0]
    safe = f"{lbl}__{base}.npy"
    cache_file = os.path.join(cache_dir, safe)
    try:
        if os.path.exists(cache_file):
            feat = np.load(cache_file)
        else:
            feat = extract_features(p, resize=resize, use_hog=use_hog)
            np.save(cache_file, feat)
        return feat, lbl, p
    except Exception as e:
        print("skip", p, e)
        return None, None, None

def cache_features_parallel(paths_labels, cache_dir="cache_aug", resize=(128,128), use_hog=False, n_jobs=None):
    os.makedirs(cache_dir, exist_ok=True)
    if n_jobs is None:
        n_jobs = max(1, min(8, cpu_count()-1))
    results = Parallel(n_jobs=n_jobs, verbose=5, prefer="threads")(
        delayed(_proc_and_cache)(tpl, cache_dir, resize, use_hog) for tpl in paths_labels
    )
    feats, labels, paths = [], [], []
    for f,l,p in results:
        if f is None: continue
        feats.append(f); labels.append(l); paths.append(p)
    X = np.vstack(feats) if len(feats)>0 else np.zeros((0,0))
    y = np.array(labels)
    return X, y, paths

def train_stage1_xgboost_pipeline(X_train, y_train, X_val, y_val, use_pca=True, pca_var=0.90, calibrate=True, save_path='stage1_xgb_augmented.joblib'):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    print("Label distribution (train):", Counter(y_train))
    print("Label distribution (val):", Counter(y_val))
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_val)
    pca = None
    if use_pca:
        pca = PCA(n_components=pca_var, svd_solver='full', random_state=42)
        Xtr = pca.fit_transform(Xtr)
        Xv = pca.transform(Xv)
        print("PCA -> components:", Xtr.shape[1])
    n_classes = len(np.unique(y_train_enc))
    objective = 'binary:logistic' if n_classes==2 else 'multi:softprob'
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, use_label_encoder=False, verbosity=0, objective=objective, random_state=42)
    print("Training XGBoost...")
    xgb.fit(Xtr, y_train_enc)
    final = xgb
    if calibrate:
        print("Calibrating probabilities...")
        cal = CalibratedClassifierCV(xgb, cv=3, method='isotonic')
        cal.fit(Xtr, y_train_enc)
        final = cal
    ypred_enc = final.predict(Xv)
    try:
        ypred = le.inverse_transform(np.asarray(ypred_enc).astype(int))
    except Exception:
        ypred = ypred_enc
    print("Validation report (after augment & retrain):")
    print(classification_report(y_val, ypred))
    joblib.dump({'scaler':scaler, 'pca':pca, 'model':final, 'label_encoder':le}, save_path)
    print("Saved pipeline:", save_path)
    return save_path

def find_best_threshold(pipeline_path, X_val, y_val, positive_class='Non-Organik'):
    pipe = joblib.load(pipeline_path)
    scaler, pca, model = pipe['scaler'], pipe.get('pca', None), pipe['model']
    le = pipe.get('label_encoder', None)
    Xv = scaler.transform(X_val)
    if pca is not None: Xv = pca.transform(Xv)
    probs = model.predict_proba(Xv)
    if le is not None:
        class_names = list(le.inverse_transform(np.arange(len(le.classes_))))
    else:
        class_names = list(model.classes_)
    idx_pos = class_names.index(positive_class)
    best = (0.5, -1)
    for thr in np.linspace(0.01, 0.99, 99):
        argmax_idx = np.argmax(probs, axis=1)
        argmax_name = [class_names[i] for i in argmax_idx]
        preds = [positive_class if prob[idx_pos] >= thr else argmax_name[i] for i, prob in enumerate(probs)]
        f1 = f1_score(y_val, preds, average='macro')
        if f1 > best[1]:
            best = (thr, f1)
    cfg = {'threshold': float(best[0]), 'f1': float(best[1]), 'positive_class': positive_class}
    with open('stage1_threshold.json','w') as f:
        json.dump(cfg, f)
    print("Saved stage1_threshold.json:", cfg)
    return best

def gather_paths(root):
    root = root.rstrip('/')
    org_dir = os.path.join(root, 'Organic')
    nonorg_dir = os.path.join(root, 'Non-Organik')
    pairs = []
    if os.path.isdir(org_dir):
        for sub in os.listdir(org_dir):
            sp = os.path.join(org_dir, sub)
            if os.path.isdir(sp):
                for f in os.listdir(sp):
                    pairs.append((os.path.join(sp,f), 'Organic'))
    if os.path.isdir(nonorg_dir):
        for sub in os.listdir(nonorg_dir):
            sp = os.path.join(nonorg_dir, sub)
            if os.path.isdir(sp):
                for f in os.listdir(sp):
                    pairs.append((os.path.join(sp,f), 'Non-Organik'))
    return pairs

def main(args):
    root = args.root
    augs_per = args.augs
    resize = (args.resize, args.resize)
    n_jobs = args.n_jobs if args.n_jobs>0 else max(1, cpu_count()-1)
    random.seed(42)
    os.makedirs(root, exist_ok=True)
    print("Gathering dataset paths...")
    pairs = gather_paths(root)
    organic_paths = [p for p,l in pairs if l=='Organic']
    print("Found Organic images:", len(organic_paths))
    if len(organic_paths)==0:
        print("No Organic images found. Exiting.")
        return

    aug_root = os.path.join(root, 'Organic_aug')
    if os.path.exists(aug_root):
        print("Note: Organic_aug exists. Old augmentations will be reused/kept.")
    else:
        os.makedirs(aug_root, exist_ok=True)

    total_created = 0
    for p in organic_paths:
        try:
            img = cv2.imread(p)
            if img is None: 
                continue
            base = os.path.splitext(os.path.basename(p))[0]
            for k in range(augs_per):
                aug_img = augment_image(img, seed=random.randint(0,999999))
                out_name = f"{base}_aug{str(k+1)}.jpg"
                out_path = os.path.join(aug_root, out_name)
                if not os.path.exists(out_path):
                    cv2.imwrite(out_path, aug_img)
                    total_created += 1
        except Exception as e:
            print("aug error", p, e)
    print("Total augmented images created:", total_created, "in", aug_root)
    new_pairs = []
    for f in os.listdir(aug_root):
        new_pairs.append((os.path.join(aug_root,f), 'Organic'))
    for p in organic_paths:
        new_pairs.append((p, 'Organic'))
    nonorg_dir = os.path.join(root, 'Non-Organik')
    if os.path.isdir(nonorg_dir):
        for sub in os.listdir(nonorg_dir):
            sp = os.path.join(nonorg_dir, sub)
            if os.path.isdir(sp):
                for f in os.listdir(sp):
                    new_pairs.append((os.path.join(sp,f), 'Non-Organik'))

    print("Total stage1 samples after augmentation:", len(new_pairs))

    cache_dir = os.path.join('cache_aug_stage1')
    print("Caching features (threaded)...")
    X, y, paths = cache_features_parallel(new_pairs, cache_dir=cache_dir, resize=resize, use_hog=False, n_jobs=n_jobs)
    print("Cached features shapes:", X.shape, y.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    print("Stage1 shapes:", X_train.shape, X_val.shape)
    pipe_path = train_stage1_xgboost_pipeline(X_train, y_train, X_val, y_val, use_pca=True, pca_var=0.90, calibrate=True, save_path='stage1_xgb_augmented.joblib')
    best = find_best_threshold(pipe_path, X_val, y_val, positive_class='Non-Organik')
    print("Best thr:", best)
    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=str, default='garbage-dataset', help='root dataset folder')
    p.add_argument('--augs', type=int, default=3, help='augmentations per organic image')
    p.add_argument('--resize', type=int, default=128, help='resize size for feature extraction (square)')
    p.add_argument('--n_jobs', type=int, default=6, help='n_jobs for parallel caching (threads)')
    args = p.parse_args()
    main(args)
