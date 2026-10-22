
import streamlit as st
from tensorflow import keras
import numpy as np
from PIL import Image, UnidentifiedImageError

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DermAI | Skin Lesion Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

IMG_SIZE = 224
MODEL_PATH = "models/efficientnet_b0_best.keras"

CLASS_NAMES = [
    "Melanoma",
    "Nevus",
    "Seborrheic Keratosis"
]

MIN_IMAGE_SIZE = 100
MAX_IMAGE_PIXELS = 20_000_000
MAX_FILE_SIZE = 15 * 1024 * 1024
CLOSE_SCORE_THRESHOLD = 10.0


# ---------------- CUSTOM CSS ----------------
st.markdown("""
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

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

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

.info-card {
    background: #ffffff;
    border: 1px solid #e0ebe7;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 16px;
}

.result-card {
    background: #ffffff;
    border: 1px solid #dceae5;
    border-radius: 20px;
    padding: 25px;
    margin: 12px 0;
}

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

.disclaimer {
    background: #fff4d6;
    border: 1px solid #e7bd58;
    border-radius: 14px;
    padding: 16px 18px;
    color: #654b0b;
    font-size: 13px;
    line-height: 1.7;
    margin: 18px 0;
}

.disclaimer b {
    color: #503900;
}

.warning-card {
    background: #fff4d6;
    border: 1px solid #e7bd58;
    border-radius: 14px;
    padding: 16px 18px;
    color: #654b0b;
    font-size: 14px;
    line-height: 1.7;
    margin: 16px 0;
}

.warning-card h4 {
    color: #503900;
    margin: 0 0 8px 0;
}

.warning-card p {
    color: #654b0b;
    margin: 0;
}

.success-card {
    background: #eaf6ef;
    border: 1px solid #b8ddc5;
    border-radius: 14px;
    padding: 16px 18px;
    color: #245c39;
    line-height: 1.7;
    margin: 16px 0;
}

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

div[data-testid="stImage"] img {
    border-radius: 15px;
}

.stButton button {
    border-radius: 10px;
    font-weight: 700;
}

/* Improve Streamlit alert text readability */
div[data-testid="stAlert"] {
    border-radius: 14px;
}

div[data-testid="stAlert"] p {
    color: #183b3a !important;
}

@media (max-width: 700px) {
    .hero {
        padding: 25px 20px;
    }

    .result-card {
        padding: 18px;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">AI · Computer Vision · Research Prototype</div>
    <h1>DermAI</h1>
    <p>
        An AI-powered research prototype for classifying skin-lesion
        images into three predefined categories using deep learning.
        Upload an image to explore the model output and general safety
        guidance.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    <b>Research prototype — not a medical diagnosis.</b><br>
    Predictions may be incorrect. DermAI cannot diagnose or rule out
    skin cancer and has not been validated for clinical risk assessment.
    Do not use its results to decide whether a lesion is harmless or
    whether medication is needed. Consult a qualified healthcare
    professional about concerning skin changes.
</div>
""", unsafe_allow_html=True)


# ---------------- UPLOAD SECTION ----------------
st.markdown(
    '<div class="section-heading">Image analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Upload a JPG, JPEG, or PNG image to view the model output.'
    '</div>',
    unsafe_allow_html=True
)

# Inform users about supported classes before upload.
st.markdown("""
<div class="warning-card">
    <h4>Important: Supported images only</h4>
    <p>
        This model was trained on Melanoma, Nevus, and Seborrheic
        Keratosis. It is not trained to recognize normal skin or all
        other skin conditions. Basic file checks cannot determine
        whether an image contains a skin lesion.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a skin-lesion image",
    type=["jpg", "jpeg", "png"],
    help="Use a clear image similar to the ISIC dataset."
)


if uploaded_file is not None:
    try:
        # ---------------- BASIC FILE VALIDATION ----------------
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("The image is too large. Please upload an image smaller than 15 MB.")
            st.stop()

        # Verify the file can be decoded as an image.
        with Image.open(uploaded_file) as check_image:
            check_image.verify()

        # Reopen after verification.
        uploaded_file.seek(0)

        with Image.open(uploaded_file) as source_image:
            width, height = source_image.size

            if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                st.error(
                    "Image resolution is too small. Upload a clearer "
                    "image with at least 100 pixels in both dimensions."
                )
                st.stop()

            if width * height > MAX_IMAGE_PIXELS:
                st.error(
                    "Image dimensions are too large to process. "
                    "Please upload a smaller image."
                )
                st.stop()

            image = source_image.convert("RGB")

        # Warn about unusual image shapes.
        aspect_ratio = width / height

        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            st.warning(
                "This image has an unusual aspect ratio. Check that "
                "the full area of interest is visible."
            )

        st.info(
            "Basic image checks passed. These checks do not confirm "
            "that the image contains a skin lesion."
        )

        # ---------------- IMAGE PREVIEW ----------------
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown(
                '<div class="section-heading">Uploaded image</div>',
                unsafe_allow_html=True
            )
            st.image(image, use_container_width=True)
            st.caption(f"Image dimensions: {width} × {height} pixels")

        # ---------------- MODEL PREDICTION ----------------
        with right:
            st.markdown(
                '<div class="section-heading">AI model results</div>',
                unsafe_allow_html=True
            )

            with st.spinner("Loading model and analyzing image..."):
                model = load_model()

                processed_image = image.resize((IMG_SIZE, IMG_SIZE))
                image_array = np.asarray(
                    processed_image, dtype=np.float32
                )
                image_array = np.expand_dims(image_array, axis=0)

                prediction = model.predict(
                    image_array, verbose=0
                )[0]

            # Validate model output.
            if (
                prediction.shape != (len(CLASS_NAMES),)
                or not np.all(np.isfinite(prediction))
                or np.any(prediction < 0)
                or np.any(prediction > 1)
            ):
                raise ValueError("Unexpected model output.")

            # ---------------- MAIN PREDICTION ----------------
            predicted_index = int(np.argmax(prediction))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(prediction[predicted_index]) * 100

            # ---------------- CLOSE-SCORE WARNING ----------------
            sorted_scores = np.sort(prediction)
            score_gap = float(
                sorted_scores[-1] - sorted_scores[-2]
            ) * 100

            prediction_uncertain = (
                score_gap <= CLOSE_SCORE_THRESHOLD
            )

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Main predicted class</div>
                <div class="result-class">{predicted_class}</div>
                <p style="color:#68817c; margin-bottom:0;">
                    Highest-scoring class from the three-class model.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                '<div class="metric-label">'
                'Model confidence for selected class'
                '</div>',
                unsafe_allow_html=True
            )

            st.progress(float(prediction[predicted_index]))
            st.markdown(f"**{confidence:.2f}%**")

            # Display close-score warning inside the try block.
            if prediction_uncertain:
                st.markdown(f"""
                <div class="warning-card">
                    <h4>Prediction uncertainty warning</h4>
                    <p>
                        The two highest model scores differ by only
                        {score_gap:.2f} percentage points. The model's
                        output may be uncertain. This is a display
                        warning, not a validated medical-risk
                        assessment. Do not rely on this result to
                        make medical decisions.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        # ---------------- CLASS PROBABILITIES ----------------
        st.markdown("---")
        st.markdown(
            '<div class="section-heading">Class probabilities</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'These are model output scores, not medical risk percentages '
            'or proof that a lesion is harmless.'
            '</div>',
            unsafe_allow_html=True
        )

        probability_columns = st.columns(3)

        for i, class_name in enumerate(CLASS_NAMES):
            probability = float(prediction[i])

            with probability_columns[i]:
                st.markdown(
                    '<div class="info-card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="metric-label">{class_name}</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<h3 style="color:#087f70; margin:8px 0;">'
                    f'{probability * 100:.2f}%</h3>',
                    unsafe_allow_html=True
                )

                st.progress(probability)

                if i == predicted_index:
                    st.caption("Highest model output")

                st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- SAFETY INTERPRETATION ----------------
        st.markdown("---")
        st.markdown(
            '<div class="section-heading">Safety interpretation</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'The model cannot establish actual medical risk. '
            'The following is general safety guidance.'
            '</div>',
            unsafe_allow_html=True
        )

        if predicted_class == "Melanoma":
            st.markdown("""
            <div class="warning-card">
                <h4>Professional assessment is important</h4>
                <p>
                    The model selected Melanoma as its highest-scoring
                    class. This is not a diagnosis, but the result
                    should not be ignored. Arrange prompt assessment
                    with a qualified healthcare professional.
                </p>
            </div>
            """, unsafe_allow_html=True)

        elif predicted_class == "Nevus":
            st.info(
                "Model output: Nevus. This does not confirm that the "
                "lesion is benign or rule out skin cancer. Seek "
                "professional assessment if the spot is new, changing, "
                "unusual, bleeding, or otherwise concerning."
            )

        else:
            st.info(
                "Model output: Seborrheic Keratosis. This does not "
                "confirm that the lesion is benign or rule out another "
                "condition. Have a healthcare professional assess "
                "any new, changing, unusual, or concerning lesion."
            )

        # ---------------- IMAGE SUITABILITY ----------------
        st.markdown("""
        <div class="info-card">
            <h3 style="color:#087f70; margin-top:0;">
                Image suitability
            </h3>
            <p style="color:#526e69; line-height:1.8;">
                The application checks file readability, resolution,
                and dimensions. These checks do not determine whether
                an image contains a skin lesion. The model may still
                produce a class for normal skin, unrelated, blurry,
                or unsupported images.
            </p>
            <p style="color:#526e69; line-height:1.8;">
                Reliable rejection of unsuitable images requires a
                separately developed and validated detection method.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ---------------- DOCTOR GUIDANCE ----------------
        st.markdown("""
        <div class="info-card">
            <h3 style="color:#087f70; margin-top:0;">
                When should you consult a doctor?
            </h3>
            <p style="color:#526e69; line-height:1.8;">
                Arrange a medical assessment if a spot is new, changes
                in size, shape or colour, has irregular borders, looks
                different from your other spots, bleeds, forms a sore
                that does not heal, or causes persistent itching.
            </p>
            <p style="color:#526e69; line-height:1.8;">
                Do not delay evaluation because of a reassuring AI
                prediction. A healthcare professional can decide
                whether further examination or a biopsy is needed.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ---------------- TREATMENT SAFETY ----------------
        st.markdown("""
        <div class="disclaimer">
            <b>Treatment safety</b><br>
            DermAI cannot determine whether ointments, creams or tablets
            are appropriate. Do not start, stop, or substitute medication
            based on this prediction. A qualified healthcare professional
            should assess the lesion and recommend treatment if needed.
        </div>
        """, unsafe_allow_html=True)

        # ---------------- MODEL INFORMATION ----------------
        st.markdown("""
        <div class="info-card">
            <h3 style="color:#087f70; margin-top:0;">
                About this AI model
            </h3>
            <p style="color:#526e69; line-height:1.8;">
                <b>Architecture:</b> EfficientNet-B0 with transfer learning<br>
                <b>Dataset:</b> ISIC 2017<br>
                <b>Classes:</b> Melanoma, Nevus and Seborrheic Keratosis<br>
                <b>Validation accuracy:</b> 60.25% on 400 validation images
            </p>
            <p style="color:#526e69; line-height:1.8;">
                This is an educational research prototype. It has not
                been validated for clinical diagnosis, medical risk
                scoring, or triage.
            </p>
        </div>
        """, unsafe_allow_html=True)

    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(
            "The image could not be processed. Please upload a valid, "
            "clear JPG, JPEG or PNG image."
        )

    except Exception as error:
        st.error(
            "An unexpected error occurred during image analysis. "
            "Check that the model exists at "
            "models/efficientnet_b0_best.keras."
        )
        st.caption(f"Technical details: {error}")

else:
    # ---------------- EMPTY STATE ----------------
    st.markdown("""
    <div class="info-card">
        <h3 style="color:#087f70; margin-top:0;">How it works</h3>
        <p style="color:#607a75; line-height:1.8;">
            <b>01</b> &nbsp; Upload a skin-lesion image<br>
            <b>02</b> &nbsp; The app checks basic image properties<br>
            <b>03</b> &nbsp; The image is resized to 224 × 224 pixels<br>
            <b>04</b> &nbsp; EfficientNet-B0 generates three class scores<br>
            <b>05</b> &nbsp; View the main prediction and safety guidance
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <b>DermAI</b> · EfficientNet-B0 · ISIC 2017<br>
    Educational and research use only · Not for medical diagnosis
</div>
""", unsafe_allow_html=True)
