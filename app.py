
import json
import os
import io
import csv
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageStat

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Lung Cancer AI Screening",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS / SETTINGS
# ============================================================
MODEL_PATH = "lung_cancer_cnn.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)
HISTORY_FILE = "prediction_history.csv"

# Demo/local login credentials.
# Change these before demonstrating if you want custom credentials.
USERS = {
    "admin": "admin123",
    "doctor": "doctor123",
    "student": "student123",
}

# ============================================================
# CUSTOM UI
# ============================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f7fbff 0%, #eef7f8 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #063b4c 0%, #087f8c 100%);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .hero {
        padding: 28px 32px;
        border-radius: 22px;
        background: linear-gradient(135deg, #063b4c, #087f8c);
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(0,0,0,.10);
    }

    .hero h1 {
        margin: 0;
        font-size: 38px;
    }

    .hero p {
        margin: 8px 0 0;
        font-size: 17px;
        opacity: .92;
    }

    .card {
        background: white;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 6px 22px rgba(0,0,0,.07);
        border: 1px solid #e6eef2;
        margin-bottom: 16px;
    }

    .result-card {
        border-radius: 20px;
        padding: 24px;
        background: white;
        box-shadow: 0 8px 28px rgba(0,0,0,.10);
        border: 2px solid #087f8c;
    }

    .scan-id {
        font-size: 14px;
        color: #60727a;
        font-weight: 600;
    }

    .big-result {
        font-size: 34px;
        font-weight: 800;
        color: #063b4c;
    }

    .login-box {
        max-width: 520px;
        margin: 70px auto;
        padding: 35px;
        background: white;
        border-radius: 24px;
        box-shadow: 0 12px 40px rgba(0,0,0,.12);
        border: 1px solid #e2edf0;
    }

    .small-muted {
        color: #66777e;
        font-size: 13px;
    }

    .metric-card {
        background: white;
        border-radius: 17px;
        padding: 18px;
        border: 1px solid #e6eef2;
        box-shadow: 0 5px 18px rgba(0,0,0,.06);
    }

    .footer {
        text-align: center;
        color: #718088;
        padding: 20px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                names = json.load(f)
            if isinstance(names, list) and len(names) >= 3:
                return names
        except Exception:
            pass
    return ["Benign", "Malignant", "Normal"]


# ============================================================
# HISTORY
# ============================================================
HISTORY_COLUMNS = [
    "username",
    "scan_id",
    "patient_id",
    "age",
    "gender",
    "notes",
    "filename",
    "timestamp",
    "prediction",
    "confidence",
    "normal_probability",
    "benign_probability",
    "malignant_probability",
]


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    try:
        df = pd.read_csv(HISTORY_FILE)
        for col in HISTORY_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[HISTORY_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=HISTORY_COLUMNS)


def save_history(record):
    df = load_history()
    new_row = pd.DataFrame([record])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)


def clear_user_history(username):
    df = load_history()
    if df.empty:
        return
    df = df[df["username"].astype(str) != str(username)]
    df.to_csv(HISTORY_FILE, index=False)


# ============================================================
# HELPERS
# ============================================================
def make_scan_id():
    return "SCAN-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def image_quality(image):
    arr = np.asarray(image.convert("L"), dtype=np.float32)

    brightness = float(arr.mean())
    contrast = float(arr.std())
    min_intensity = float(arr.min())
    max_intensity = float(arr.max())

    return {
        "width": image.width,
        "height": image.height,
        "brightness": brightness,
        "contrast": contrast,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "mode": image.mode,
        "aspect_ratio": image.width / image.height if image.height else 0,
    }


def predict_image(model, image, class_names):
    resized = image.resize(IMG_SIZE)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)

    probabilities = model.predict(array, verbose=0)[0]
    index = int(np.argmax(probabilities))
    label = class_names[index]
    confidence = float(probabilities[index]) * 100

    return label, confidence, probabilities, resized


def generate_report_text(record):
    return f"""
LUNG CANCER AI SCREENING REPORT
================================

Scan ID       : {record['scan_id']}
Patient ID    : {record['patient_id'] or 'Not provided'}
Age           : {record['age'] or 'Not provided'}
Gender        : {record['gender'] or 'Not provided'}
Date & Time   : {record['timestamp']}
Image         : {record['filename']}

PREDICTION
----------
Result        : {record['prediction']}
Confidence    : {float(record['confidence']):.2f}%

CLASS PROBABILITIES
-------------------
Normal        : {float(record['normal_probability']):.2f}%
Benign        : {float(record['benign_probability']):.2f}%
Malignant     : {float(record['malignant_probability']):.2f}%

Notes
-----
{record['notes'] or 'None'}

DISCLAIMER
----------
This application is intended for educational/research use only.
It is not a medical diagnostic device and must not be used as a
substitute for professional medical diagnosis or clinical decisions.
"""


# ============================================================
# LOGIN
# ============================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = None


def login_page():
    st.markdown("""
    <div class="login-box">
        <h1 style="text-align:center;color:#063b4c;">🫁 Lung Cancer AI</h1>
        <p style="text-align:center;color:#68777d;">
            CNN-based CT scan analysis system
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("👤 Username")
        password = st.text_input("🔒 Password", type="password")
        submitted = st.form_submit_button(
            "Login",
            type="primary",
            use_container_width=True
        )

    if submitted:
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful.")
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.info(
        "Demo accounts: admin / admin123, doctor / doctor123, "
        "student / student123"
    )


if not st.session_state.logged_in:
    login_page()
    st.stop()

# ============================================================
# LOAD MODEL
# ============================================================
model = load_model()
class_names = load_class_names()

if model is None:
    st.error("Trained model not found.")
    st.info("First run: python prepare_dataset.py, then python train.py")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🫁 Lung Cancer AI")
    st.caption(f"Logged in as **{st.session_state.username}**")
    st.divider()

    pages = [
        "Dashboard",
        "🔍 New CT Scan",
        "📋 Scan History",
        "📊 Analytics",
    ]

    selected = st.radio(
        "Navigation",
        pages,
        index=pages.index(st.session_state.page)
        if st.session_state.page in pages else 0,
    )

    st.session_state.page = selected

    st.divider()

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.current_prediction = None
        st.rerun()

    st.caption("Educational / research application")

# ============================================================
# USER HISTORY
# ============================================================
all_history = load_history()
user_history = all_history[
    all_history["username"].astype(str)
    == str(st.session_state.username)
].copy()

# ============================================================
# DASHBOARD
# ============================================================
if st.session_state.page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>🫁 Lung Cancer AI Screening Dashboard</h1>
        <p>Analyze lung CT images using your trained CNN model.</p>
    </div>
    """, unsafe_allow_html=True)

    total = len(user_history)
    normal_count = int(
        (user_history["prediction"].astype(str).str.lower() == "normal").sum()
    )
    benign_count = int(
        (user_history["prediction"].astype(str).str.lower() == "benign").sum()
    )
    malignant_count = int(
        (user_history["prediction"].astype(str).str.lower() == "malignant").sum()
    )

    avg_conf = (
        pd.to_numeric(user_history["confidence"], errors="coerce").mean()
        if total else 0
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("🖼️ Total Scans", total)
    with c2:
        st.metric("🟢 Normal", normal_count)
    with c3:
        st.metric("🟡 Benign", benign_count)
    with c4:
        st.metric("🔴 Malignant", malignant_count)
    with c5:
        st.metric("🎯 Avg. Confidence", f"{avg_conf:.1f}%")

    st.markdown("### ⚡ Quick Actions")

    a1, a2, a3 = st.columns(3)

    with a1:
        if st.button(
            "🔍 Start New CT Scan",
            use_container_width=True,
            type="primary"
        ):
            st.session_state.page = "🔍 New CT Scan"
            st.rerun()

    with a2:
        if st.button("📋 View Scan History", use_container_width=True):
            st.session_state.page = "📋 Scan History"
            st.rerun()

    with a3:
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.page = "📊 Analytics"
            st.rerun()

    st.markdown("### 🕘 Recent Scans")

    if user_history.empty:
        st.info("No scans have been analyzed yet. Start your first CT scan.")
    else:
        recent = user_history.tail(5).iloc[::-1].copy()
        recent_display = recent[
            [
                "scan_id",
                "patient_id",
                "timestamp",
                "prediction",
                "confidence",
            ]
        ].copy()
        recent_display["confidence"] = pd.to_numeric(
            recent_display["confidence"], errors="coerce"
        ).map(lambda x: f"{x:.2f}%")

        st.dataframe(
            recent_display,
            use_container_width=True,
            hide_index=True,
        )

    st.warning(
        "Educational/research use only. This system is not a medical "
        "diagnostic device."
    )

# ============================================================
# NEW CT SCAN
# ============================================================
elif st.session_state.page == "🔍 New CT Scan":

    st.markdown("""
    <div class="hero">
        <h1>🔍 New CT Scan Analysis</h1>
        <p>Upload a lung CT image and run CNN-based classification.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 👤 Scan Information")

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        patient_id = st.text_input(
            "Patient ID (optional)",
            placeholder="e.g. P001"
        )

    with p2:
        age = st.number_input(
            "Age (optional)",
            min_value=0,
            max_value=120,
            value=0,
        )

    with p3:
        gender = st.selectbox(
            "Gender",
            ["Not specified", "Female", "Male", "Other"]
        )

    with p4:
        scan_id = st.text_input(
            "Scan ID",
            value=make_scan_id()
        )

    notes = st.text_area(
        "📝 Scan Notes (optional)",
        placeholder="Add any notes related to this scan..."
    )

    st.markdown("### 🖼️ Upload CT Scan")

    uploaded_file = st.file_uploader(
        "Drag and drop a CT scan image here",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported image formats: JPG, JPEG, PNG, BMP and WEBP.",
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        quality = image_quality(image)

        left, right = st.columns([1.35, 1])

        with left:
            st.image(
                image,
                caption="Uploaded CT Scan",
                use_container_width=True,
            )

        with right:
            st.markdown("### 🧪 Image Quality")

            q1, q2 = st.columns(2)

            with q1:
                st.metric("Width", f"{quality['width']} px")
                st.metric("Brightness", f"{quality['brightness']:.1f}")

            with q2:
                st.metric("Height", f"{quality['height']} px")
                st.metric("Contrast", f"{quality['contrast']:.1f}")

            st.write(
                f"**Aspect ratio:** {quality['aspect_ratio']:.2f}"
            )
            st.write(
                f"**Intensity range:** "
                f"{quality['min_intensity']:.0f} – "
                f"{quality['max_intensity']:.0f}"
            )
            st.write(f"**Image mode:** {quality['mode']}")
            st.write(
                f"**File size:** "
                f"{uploaded_file.size / 1024:.1f} KB"
            )

        st.divider()

        if st.button(
            "🧠 Analyze CT Scan",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner("CNN is analyzing the CT image..."):
                try:
                    label, confidence, probabilities, resized = predict_image(
                        model, image, class_names
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.stop()

            # Map probabilities by class name
            probability_map = {
                str(name).lower(): float(prob)
                for name, prob in zip(class_names, probabilities)
            }

            normal_prob = probability_map.get("normal", 0.0) * 100
            benign_prob = probability_map.get("benign", 0.0) * 100
            malignant_prob = probability_map.get("malignant", 0.0) * 100

            record = {
                "username": st.session_state.username,
                "scan_id": scan_id,
                "patient_id": patient_id,
                "age": "" if age == 0 else age,
                "gender": "" if gender == "Not specified" else gender,
                "notes": notes,
                "filename": uploaded_file.name,
                "timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "prediction": label,
                "confidence": confidence,
                "normal_probability": normal_prob,
                "benign_probability": benign_prob,
                "malignant_probability": malignant_prob,
            }

            save_history(record)
            st.session_state.current_prediction = record

            st.success("CT scan analysis completed.")

            st.markdown("### 🎯 Prediction Result")

            r1, r2 = st.columns([1.2, 1])

            with r1:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="scan-id">{scan_id}</div>
                        <div class="big-result">{label}</div>
                        <p>Predicted class</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with r2:
                st.metric(
                    "🎯 Confidence",
                    f"{confidence:.2f}%"
                )

            st.markdown("### 📊 Class Probabilities")

            prob_values = {
                "Normal": normal_prob,
                "Benign": benign_prob,
                "Malignant": malignant_prob,
            }

            for name, value in prob_values.items():
                st.write(f"**{name} — {value:.2f}%**")
                st.progress(
                    min(max(value / 100, 0.0), 1.0)
                )

            st.markdown("### 🔬 CNN Preprocessing Preview")

            x1, x2 = st.columns(2)

            with x1:
                st.image(
                    image,
                    caption="Original CT image",
                    use_container_width=True,
                )

            with x2:
                st.image(
                    resized,
                    caption="224 × 224 image supplied to CNN",
                    use_container_width=True,
                )

            st.markdown("### 📄 Prediction Report")

            report_text = generate_report_text(record)

            st.download_button(
                "⬇️ Download Scan Report",
                data=report_text,
                file_name=f"{scan_id}_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

            st.info(
                "The prediction is generated by the trained CNN model. "
                "It should not be interpreted as a clinical diagnosis."
            )

# ============================================================
# SCAN HISTORY
# ============================================================
elif st.session_state.page == "📋 Scan History":

    st.markdown("""
    <div class="hero">
        <h1>📋 Scan History</h1>
        <p>Review and export scans analyzed in this session/application.</p>
    </div>
    """, unsafe_allow_html=True)

    if user_history.empty:
        st.info("No scan records found.")
    else:
        f1, f2 = st.columns(2)

        with f1:
            search = st.text_input(
                "🔎 Search",
                placeholder="Scan ID, patient ID or filename"
            )

        with f2:
            prediction_filter = st.selectbox(
                "Filter prediction",
                ["All", "Normal", "Benign", "Malignant"]
            )

        filtered = user_history.copy()

        if search:
            mask = (
                filtered["scan_id"].astype(str).str.contains(
                    search, case=False, na=False
                )
                | filtered["patient_id"].astype(str).str.contains(
                    search, case=False, na=False
                )
                | filtered["filename"].astype(str).str.contains(
                    search, case=False, na=False
                )
            )
            filtered = filtered[mask]

        if prediction_filter != "All":
            filtered = filtered[
                filtered["prediction"].astype(str).str.lower()
                == prediction_filter.lower()
            ]

        display = filtered[
            [
                "scan_id",
                "patient_id",
                "timestamp",
                "prediction",
                "confidence",
                "filename",
            ]
        ].copy()

        display["confidence"] = pd.to_numeric(
            display["confidence"], errors="coerce"
        ).map(lambda x: f"{x:.2f}%")

        st.dataframe(
            display.iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### 📥 Export Data")

        csv_data = filtered.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download History as CSV",
            data=csv_data,
            file_name="lung_cancer_prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("### 🧹 History Management")

        if st.button(
            "🗑️ Clear My Scan History",
            use_container_width=True
        ):
            clear_user_history(st.session_state.username)
            st.session_state.current_prediction = None
            st.success("Your scan history has been cleared.")
            st.rerun()

# ============================================================
# ANALYTICS
# ============================================================
elif st.session_state.page == "📊 Analytics":

    st.markdown("""
    <div class="hero">
        <h1>📊 Scan Analytics</h1>
        <p>Visual summary of your stored CT scan predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    if user_history.empty:
        st.info("Analyze some CT scans first to generate analytics.")
    else:
        user_history["confidence_num"] = pd.to_numeric(
            user_history["confidence"], errors="coerce"
        )

        counts = (
            user_history["prediction"]
            .astype(str)
            .value_counts()
            .reindex(
                ["Normal", "Benign", "Malignant"],
                fill_value=0
            )
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 📈 Prediction Distribution")
            st.bar_chart(counts)

        with c2:
            st.markdown("### 🎯 Confidence by Scan")
            conf_chart = user_history[
                ["scan_id", "confidence_num"]
            ].set_index("scan_id")
            st.line_chart(conf_chart)

        st.markdown("### 📊 Summary")

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.metric("Total Scans", len(user_history))

        with s2:
            st.metric(
                "Average Confidence",
                f"{user_history['confidence_num'].mean():.2f}%"
            )

        with s3:
            st.metric(
                "Highest Confidence",
                f"{user_history['confidence_num'].max():.2f}%"
            )

        with s4:
            st.metric(
                "Lowest Confidence",
                f"{user_history['confidence_num'].min():.2f}%"
            )

        st.markdown("### 🧾 Detailed Probability Data")

        details = user_history[
            [
                "scan_id",
                "timestamp",
                "prediction",
                "confidence",
                "normal_probability",
                "benign_probability",
                "malignant_probability",
            ]
        ].copy()

        st.dataframe(
            details.iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        🫁 Lung Cancer AI Screening • CNN-based image classification<br>
        Educational / research use only — not a medical diagnostic device.
    </div>
    """,
    unsafe_allow_html=True,
)
