import csv
import hashlib
import io

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow import keras

from database import (
    init_db,
    save_prediction,
    get_prediction_history,
    get_all_prediction_history,
    delete_prediction_history,
)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="DermAI | Skin Lesion Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

IMG_SIZE = 224
MODEL_PATH = "models/efficientnet_b0_four_class_finetuned_best.keras"

# IMPORTANT: This order must match the output order used during training.
CLASS_NAMES = [
    "Melanoma",
    "Nevus",
    "Seborrheic Keratosis",
    "Normal Skin",
]

MIN_IMAGE_SIZE = 100
MAX_IMAGE_PIXELS = 60_000_000  # Safety limit for decoding uploaded images
MAX_PROCESSING_DIMENSION = 2048  # Downscale large uploads before preview/model processing
MAX_FILE_SIZE = 15 * 1024 * 1024
CLOSE_SCORE_THRESHOLD = 10.0

# ---------------------------------------------------------
# DATABASE INITIALIZATION
# ---------------------------------------------------------
try:
    init_db()
    database_available = True
except Exception as db_error:
    database_available = False
    st.warning("Prediction-history storage is unavailable.")
    st.caption(f"Database detail: {db_error}")

if "saved_prediction_hashes" not in st.session_state:
    st.session_state.saved_prediction_hashes = set()

if "confirm_clear_history" not in st.session_state:
    st.session_state.confirm_clear_history = False

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: #f5f9f8;
        color: #183b3a;
        font-family: 'DM Sans', sans-serif;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    .hero {
        background: linear-gradient(120deg, #ffffff 0%, #e7f5f1 100%);
        border: 1px solid #dcece7;
        border-radius: 24px;
        padding: 35px 38px;
        margin-bottom: 24px;
    }
    .eyebrow {
        color: #087f70;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .hero h1 {
        color: #153f3b;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(32px, 4vw, 48px);
        font-weight: 800;
        margin: 0;
        line-height: 1.15;
    }
    .hero p {
        color: #5a7773;
        font-size: 16px;
        line-height: 1.7;
        max-width: 720px;
        margin-top: 14px;
    }
    .section-heading {
        color: #183b3a;
        font-family: 'Manrope', sans-serif;
        font-size: 22px;
        font-weight: 800;
        margin: 12px 0 6px 0;
    }
    .section-subtitle {
        color: #69817d;
        font-size: 14px;
        margin-bottom: 18px;
    }
    .info-card, .result-card {
        background: #ffffff;
        border: 1px solid #dceae5;
        border-radius: 18px;
        padding: 22px;
        margin: 12px 0;
    }
    .result-card { padding: 25px; border-radius: 20px; }
    .result-label {
        color: #6a8580;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        font-weight: 700;
    }
    .result-class {
        color: #087f70;
        font-family: 'Manrope', sans-serif;
        font-size: 27px;
        font-weight: 800;
        margin-top: 7px;
    }
    .metric-label {
        color: #68817c;
        font-size: 13px;
        font-weight: 600;
    }
    .disclaimer, .warning-card {
        background: #fff4d6;
        border: 1px solid #e7bd58;
        border-radius: 14px;
        padding: 16px 18px;
        color: #654b0b;
        font-size: 14px;
        line-height: 1.7;
        margin: 16px 0;
    }
    .disclaimer b, .warning-card h4 { color: #503900; }
    .warning-card h4 { margin: 0 0 8px 0; }
    .warning-card p { color: #654b0b; margin: 0; }
    .footer {
        color: #78908b;
        text-align: center;
        font-size: 12px;
        padding: 24px 0 6px 0;
        border-top: 1px solid #e0ebe7;
        margin-top: 30px;
    }
    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #83bdb0;
        border-radius: 16px;
        padding: 12px;
    }
    div[data-testid="stImage"] img { border-radius: 15px; }
    .stButton button {
        border-radius: 10px;
        font-weight: 700;
        color: #ffffff !important;
    }
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label {
        color: #183b3a !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
    div[data-testid="stAlert"] { border-radius: 14px; }
    div[data-testid="stAlert"] p { color: #183b3a !important; }

    @media (max-width: 700px) {
        .hero { padding: 25px 20px; }
        .result-card { padding: 18px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


# ---------------------------------------------------------
# HEADER AND DISCLAIMER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI · Computer Vision · Research Prototype</div>
        <h1>DermAI</h1>
        <p>
            An AI-powered research prototype for classifying skin images
            into four predefined categories using deep learning.
            Upload an image to explore the model output and general
            safety guidance.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer">
        <b>Research prototype — not a medical diagnosis.</b><br>
        Predictions may be incorrect. DermAI cannot diagnose or rule out
        skin cancer and has not been validated for clinical risk assessment.
        Do not use its results to decide whether a lesion is harmless.
        Consult a qualified healthcare professional about concerning skin changes.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------
st.markdown(
    '<div class="section-heading">Image analysis</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Upload a JPG, JPEG, or PNG image to view the model output.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="warning-card">
        <h4>Important: Supported classes</h4>
        <p>
            This model was trained to classify images into Melanoma, Nevus,
            Seborrheic Keratosis, and Normal Skin categories. It may still
            produce incorrect results for unsupported images or skin conditions.
            Basic file checks do not confirm that an image is suitable for analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"],
    help="Use a clear image similar to the images used in the training dataset.",
)

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        image_digest = hashlib.sha256(file_bytes).hexdigest()

        # Validate file size.
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("The image is too large. Please upload an image smaller than 15 MB.")
            st.stop()

        # Verify image integrity.
        with Image.open(io.BytesIO(file_bytes)) as check_image:
            check_image.verify()

        # Reopen, orient, validate and automatically resize large images.
        with Image.open(io.BytesIO(file_bytes)) as source_image:
            original_width, original_height = source_image.size

            if original_width < MIN_IMAGE_SIZE or original_height < MIN_IMAGE_SIZE:
                st.error(
                    "Image resolution is too small. Upload an image with "
                    "at least 100 pixels in both dimensions."
                )
                st.stop()

            # Refuse only extremely large files to avoid excessive memory use.
            if original_width * original_height > MAX_IMAGE_PIXELS:
                st.error(
                    "This image is exceptionally large and cannot be safely opened. "
                    "Please resize it or upload an image below 60 megapixels."
                )
                st.stop()

            # Correct phone-camera orientation, then reduce large dimensions.
            oriented_image = ImageOps.exif_transpose(source_image)
            image = oriented_image.convert("RGB")
            image.thumbnail(
                (MAX_PROCESSING_DIMENSION, MAX_PROCESSING_DIMENSION),
                Image.Resampling.LANCZOS,
            )

            processed_width, processed_height = image.size

        aspect_ratio = original_width / original_height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            st.warning(
                "This image has an unusual aspect ratio. Check that the "
                "area of interest is visible."
            )

        st.info(
            "Basic image checks passed. These checks do not confirm that "
            "the image contains a skin lesion."
        )

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown(
                '<div class="section-heading">Uploaded image</div>',
                unsafe_allow_html=True,
            )
            st.image(image, use_container_width=True)
            if (processed_width, processed_height) != (original_width, original_height):
                st.caption(
                    f"Original: {original_width} × {original_height} pixels · "
                    f"Automatically resized to: {processed_width} × {processed_height} pixels"
                )
            else:
                st.caption(f"Image dimensions: {original_width} × {original_height} pixels")

        with right:
            st.markdown(
                '<div class="section-heading">AI model results</div>',
                unsafe_allow_html=True,
            )

            with st.spinner("Loading model and analyzing image..."):
                model = load_model()
                processed_image = image.resize((IMG_SIZE, IMG_SIZE))
                image_array = np.asarray(processed_image, dtype=np.float32)
                image_array = np.expand_dims(image_array, axis=0)
                prediction = model.predict(image_array, verbose=0)[0]

            # Confirm that the model returns four valid scores.
            if (
                prediction.shape != (4,)
                or not np.all(np.isfinite(prediction))
                or np.any(prediction < 0)
                or np.any(prediction > 1)
            ):
                raise ValueError(
                    "The model did not return four valid class scores. "
                    "Check that the selected model is the four-class model."
                )

            predicted_index = int(np.argmax(prediction))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(prediction[predicted_index]) * 100

            # Save all four class scores to SQLite.
            if database_available and image_digest not in st.session_state.saved_prediction_hashes:
                try:
                    save_prediction(
                        image_name=uploaded_file.name,
                        predicted_class=predicted_class,
                        confidence=confidence,
                        melanoma_score=float(prediction[0]) * 100,
                        nevus_score=float(prediction[1]) * 100,
                        seborrheic_keratosis_score=float(prediction[2]) * 100,
                        non_lesional_skin_score=float(prediction[3]) * 100,
                    )
                    st.session_state.saved_prediction_hashes.add(image_digest)
                    st.success("Prediction saved to local history.")
                except Exception as db_error:
                    st.warning("Prediction generated, but could not be saved to history.")
                    st.caption(f"Database detail: {db_error}")

            # Compare the two highest scores. This is only a model-output warning.
            sorted_scores = np.sort(prediction)
            score_gap = float(sorted_scores[-1] - sorted_scores[-2]) * 100

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Main predicted class</div>
                    <div class="result-class">{predicted_class}</div>
                    <p style="color:#68817c; margin-bottom:0;">
                        Highest-scoring class from the four-class model.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="metric-label">Model score for selected class</div>',
                unsafe_allow_html=True,
            )
            st.progress(float(prediction[predicted_index]))
            st.markdown(f"**{confidence:.2f}%**")

            if score_gap <= CLOSE_SCORE_THRESHOLD:
                st.markdown(
                    f"""
                    <div class="warning-card">
                        <h4>Prediction uncertainty warning</h4>
                        <p>
                            The two highest model scores differ by only
                            {score_gap:.2f} percentage points. The model output
                            may be uncertain. This is not a validated medical-risk
                            assessment. Do not rely on it for medical decisions.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # -------------------------------------------------
        # FOUR CLASS PROBABILITIES
        # -------------------------------------------------
        st.markdown("---")
        st.markdown(
            '<div class="section-heading">Class probabilities</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-subtitle">These are model output scores, not medical risk percentages or proof that skin is healthy.</div>',
            unsafe_allow_html=True,
        )

        probability_columns = st.columns(4)
        for index, class_name in enumerate(CLASS_NAMES):
            probability = float(prediction[index])
            with probability_columns[index]:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="metric-label">{class_name}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<h3 style="color:#087f70; margin:8px 0;">{probability * 100:.2f}%</h3>',
                    unsafe_allow_html=True,
                )
                st.progress(probability)
                if index == predicted_index:
                    st.caption("Highest model output")
                st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------------------------
        # SAFETY INTERPRETATION
        # -------------------------------------------------
        st.markdown("---")
        st.markdown(
            '<div class="section-heading">Safety interpretation</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-subtitle">The model cannot establish actual medical risk. This is general safety guidance only.</div>',
            unsafe_allow_html=True,
        )

        if predicted_class == "Melanoma":
            st.warning(
                "The model selected Melanoma as its highest-scoring class. "
                "This is not a diagnosis, but it should not be ignored. "
                "Arrange prompt assessment with a qualified healthcare professional."
            )
        elif predicted_class == "Nevus":
            st.info(
                "Model output: Nevus. This does not confirm that the lesion is "
                "benign or rule out skin cancer. Seek professional assessment if "
                "the spot is new, changing, unusual, bleeding, or concerning."
            )
        elif predicted_class == "Seborrheic Keratosis":
            st.info(
                "Model output: Seborrheic Keratosis. This is not a diagnosis "
                "and does not rule out another condition. Seek professional "
                "assessment for any new, changing, unusual, or concerning lesion."
            )
        else:
            st.info(
                "Model output: Normal Skin (non-lesional skin category). This "
                "does not guarantee that the skin is healthy or rule out a skin "
                "condition. Consult a healthcare professional if you notice "
                "concerning changes."
            )

        # -------------------------------------------------
        # GENERAL DOCTOR GUIDANCE
        # -------------------------------------------------
        st.markdown(
            """
            <div class="info-card">
                <h3 style="color:#087f70; margin-top:0;">When should you consult a doctor?</h3>
                <p style="color:#526e69; line-height:1.8;">
                    Arrange a medical assessment if a spot is new, changes in
                    size, shape or colour, has irregular borders, looks different
                    from your other spots, bleeds, forms a sore that does not heal,
                    or causes persistent itching.
                </p>
                <p style="color:#526e69; line-height:1.8;">
                    Do not delay evaluation because of a reassuring AI prediction.
                    A healthcare professional can decide whether further
                    examination or a biopsy is needed.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="disclaimer">
                <b>Treatment safety</b><br>
                DermAI cannot determine whether ointments, creams or tablets
                are appropriate. Do not start, stop, or substitute medication
                based on this prediction. A qualified healthcare professional
                should assess the lesion and recommend treatment if needed.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # MODEL INFORMATION
        # -------------------------------------------------
        st.markdown(
            """
            <div class="info-card">
                <h3 style="color:#087f70; margin-top:0;">About this AI model</h3>
                <p style="color:#526e69; line-height:1.8;">
                    <b>Architecture:</b> EfficientNet-B0 with transfer learning<br>
                    <b>Model:</b> Fine-tuned four-class model<br>
                    <b>Classes:</b> Melanoma, Nevus, Seborrheic Keratosis,
                    Normal Skin<br>
                    <b>Validation accuracy:</b> 77.96% on 785 validation images
                </p>
                <p style="color:#526e69; line-height:1.8;">
                    This is an educational research prototype. It has not been
                    validated for clinical diagnosis, medical risk scoring,
                    or triage.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="info-card">
                <h3 style="color:#087f70; margin-top:0;">Image suitability</h3>
                <p style="color:#526e69; line-height:1.8;">
                    The application checks file readability, resolution and
                    dimensions. These checks do not determine whether an image
                    contains a skin lesion. The model may still produce a class
                    for normal, unrelated, blurry or unsupported images.
                </p>
                <p style="color:#526e69; line-height:1.8;">
                    Reliable rejection of unsuitable images requires a separately
                    developed and validated detection method.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(
            "The image could not be processed. Please upload a valid, "
            "clear JPG, JPEG or PNG image."
        )
        st.caption(f"Technical detail: {error}")

    except Exception as error:
        st.error(
            "An unexpected error occurred during image analysis. "
            "Check that the four-class model file exists at: "
            f"{MODEL_PATH}"
        )
        st.caption(f"Technical detail: {error}")

else:
    st.markdown(
        """
        <div class="info-card">
            <h3 style="color:#087f70; margin-top:0;">How it works</h3>
            <p style="color:#607a75; line-height:1.8;">
                <b>01</b> &nbsp; Upload a skin image<br>
                <b>02</b> &nbsp; The app checks basic image properties<br>
                <b>03</b> &nbsp; The image is resized to 224 × 224 pixels<br>
                <b>04</b> &nbsp; EfficientNet-B0 generates four class scores<br>
                <b>05</b> &nbsp; View the main prediction and general safety guidance
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# PREDICTION HISTORY
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div class="section-heading">Prediction history</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">The most recent 20 locally stored predictions. Scores are model outputs, not medical risk percentages.</div>',
    unsafe_allow_html=True,
)

if st.session_state.get("history_action_message"):
    st.success(st.session_state.history_action_message)
    del st.session_state["history_action_message"]

try:
    history = get_prediction_history(limit=20)
    all_history = get_all_prediction_history()
    history_load_error = None
except Exception as db_error:
    history = []
    all_history = []
    history_load_error = db_error

if history_load_error is None:
    export_col, clear_col, _ = st.columns([1.2, 1.2, 1])

    with export_col:
        if all_history:
            csv_buffer = io.StringIO(newline="")
            csv_fields = [
                "id",
                "created_at",
                "image_name",
                "predicted_class",
                "confidence",
                "melanoma_score",
                "nevus_score",
                "seborrheic_keratosis_score",
                "non_lesional_skin_score",
            ]
            csv_writer = csv.DictWriter(
                csv_buffer,
                fieldnames=csv_fields,
                extrasaction="ignore",
            )
            csv_writer.writeheader()
            csv_writer.writerows(all_history)

            st.download_button(
                "Export history to CSV",
                data=csv_buffer.getvalue().encode("utf-8-sig"),
                file_name="dermai_prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
                help=f"Download all {len(all_history)} saved prediction records.",
            )
        else:
            st.button(
                "Export history to CSV",
                disabled=True,
                use_container_width=True,
                help="Save at least one prediction before exporting history.",
            )

    with clear_col:
        if st.button(
            "Clear all history",
            disabled=not bool(history),
            use_container_width=True,
            help="Permanently delete all saved prediction records from the local database.",
        ):
            st.session_state.confirm_clear_history = True

    if st.session_state.confirm_clear_history:
        st.warning(
            "This will permanently delete all prediction records from the local "
            "SQLite database. This action cannot be undone."
        )
        confirm_col, cancel_col, _ = st.columns([1, 1, 2])

        with confirm_col:
            if st.button(
                "Yes, delete all history",
                type="primary",
                use_container_width=True,
            ):
                try:
                    deleted_count = delete_prediction_history()
                    st.session_state.confirm_clear_history = False
                    st.session_state.history_action_message = (
                        f"Deleted {deleted_count} prediction record(s)."
                    )
                    st.session_state.saved_prediction_hashes = set()
                    st.rerun()
                except Exception as db_error:
                    st.error("Could not clear prediction history.")
                    st.caption(f"Database detail: {db_error}")

        with cancel_col:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_clear_history = False
                st.rerun()

    st.markdown("#### Find predictions")
    search_col, class_col = st.columns([2, 1])

    with search_col:
        history_search = st.text_input(
            "Search by image, predicted class, or date",
            placeholder="For example: ISIC_0000006 or Melanoma",
            key="history_search",
        ).strip().casefold()

    with class_col:
        selected_history_class = st.selectbox(
            "Filter by predicted class",
            options=["All classes"] + CLASS_NAMES,
            key="history_class_filter",
        )

    matching_history = []
    for row in all_history:
        searchable_text = " ".join(
            str(row.get(field, ""))
            for field in ("image_name", "predicted_class", "created_at")
        ).casefold()

        matches_search = not history_search or history_search in searchable_text
        matches_class = (
            selected_history_class == "All classes"
            or row.get("predicted_class") == selected_history_class
        )

        if matches_search and matches_class:
            matching_history.append(row)

    displayed_history = matching_history[:20]

    if all_history:
        st.caption(
            f"Showing {len(displayed_history)} of {len(matching_history)} matching "
            "record(s). The latest 20 matches are shown; CSV export includes all records."
        )

    if displayed_history:
        history_table = [
            {
                "Date & time": row.get("created_at", ""),
                "Image": row.get("image_name", ""),
                "Predicted class": row.get("predicted_class", ""),
                "Confidence (%)": round(float(row.get("confidence", 0)), 2),
                "Melanoma (%)": round(float(row.get("melanoma_score", 0)), 2),
                "Nevus (%)": round(float(row.get("nevus_score", 0)), 2),
                "Seborrheic Keratosis (%)": round(
                    float(row.get("seborrheic_keratosis_score", 0)), 2
                ),
                "Normal Skin (%)": round(
                    float(row.get("non_lesional_skin_score", 0)), 2
                ),
            }
            for row in displayed_history
        ]
        st.dataframe(history_table, use_container_width=True, hide_index=True)

    elif all_history:
        st.info("No saved predictions match these search and filter options.")
    else:
        st.info("No prediction history yet. Upload an image to save the first record.")

else:
    st.warning(
        "Prediction history could not be loaded. The model may still run normally."
    )
    st.caption(f"Database detail: {history_load_error}")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        <b>DermAI</b> · EfficientNet-B0 · Four-class research model<br>
        Educational and research use only · Not for medical diagnosis
    </div>
    """,
    unsafe_allow_html=True,
)
