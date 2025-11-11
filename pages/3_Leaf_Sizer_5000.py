import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Leaf Size 5000", page_icon="🍂")
def calc_area(img, kernel_size=5, blur=5, leaf_iter=2, ref_size=2):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 1️⃣ Estimate background color & brightness from border ---
    border = 20
    h, w = img.shape[:2]
    bg_samples = np.concatenate([
        img[:border, :, :].reshape(-1, 3),
        img[-border:, :, :].reshape(-1, 3),
        img[:, :border, :].reshape(-1, 3),
        img[:, -border:, :].reshape(-1, 3)
    ])

    bg_mean = np.mean(bg_samples, axis=0)
    bg_std = np.std(bg_samples, axis=0)

    bg_gray_mean = np.mean(gray[:border, :])  # approximate brightness

    # --- 2️⃣ Compute color + brightness deviation from background ---
    diff = img.astype(np.float32) - bg_mean  # shape (h, w, 3)
    color_diff = np.sqrt(np.sum(diff ** 2, axis=2))  # per-pixel Euclidean distance
    gray_diff = np.abs(gray.astype(np.float32) - bg_gray_mean)

    # Normalize for combination
    color_diff_norm = cv2.normalize(color_diff, None, 0, 255, cv2.NORM_MINMAX)
    gray_diff_norm = cv2.normalize(gray_diff, None, 0, 255, cv2.NORM_MINMAX)

    # Combine both signals (color + brightness)
    combined = cv2.addWeighted(color_diff_norm, 0.6, gray_diff_norm, 0.4, 0)

    # --- 3️⃣ Adaptive threshold ---
    combined_blur = cv2.GaussianBlur(combined.astype(np.uint8), (blur, blur), 0)
    _, mask_leaf = cv2.threshold(combined_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if the leaf appears as background
    if np.mean(gray[mask_leaf > 0]) > np.mean(gray[mask_leaf == 0]):
        mask_leaf = cv2.bitwise_not(mask_leaf)

    # --- 4️⃣ Morphological cleanup ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_leaf = cv2.morphologyEx(mask_leaf, cv2.MORPH_CLOSE, kernel, iterations=leaf_iter+1)
    mask_leaf = cv2.morphologyEx(mask_leaf, cv2.MORPH_OPEN, kernel, iterations=leaf_iter)

    cv2.imwrite("debug_mask_leaf.jpg", mask_leaf)

    # --- 5️⃣ Get largest contour ---
    contours, _ = cv2.findContours(mask_leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        leaf_area_px = max(cv2.contourArea(c) for c in contours)
    else:
        leaf_area_px = 0

    print(f"Leaf area (pixels): {leaf_area_px:.2f}")

    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])

    # Create a mask for orange areas
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Find contours from mask
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours for visual check
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    cv2.imwrite("debug_contours.jpg", img_contours)

    # Calculate areas of all contours
    areas = [(cv2.contourArea(c), c) for c in contours]
    areas = sorted(areas, key=lambda x: x[0], reverse=True)
    return ref_size * leaf_area_px/areas[0][0], mask_leaf, mask

st.title("Leaf-Sizer-5000")
st.markdown("How Big is my Leaf?")
st.sidebar.header("Leaf Sizer 5000")
col1, col2, col3 = st.columns(3)

with st.form("parameters_form"):
    st.header("Reference Size and Image")

    ref_size = st.slider("Reference Square Size (cm^2)", 1, 100, 1)

    img_f = st.file_uploader(label="Input image", type=None, accept_multiple_files=False,
                           key=None, help=None,
                           on_change=None, args=None,
                           kwargs=None, disabled=False,
                           label_visibility="visible", width="stretch")

    submitted = st.form_submit_button("Calculate Size")

if submitted:
    with st.spinner("Calculating Leaf Size..."):
        file_bytes = np.asarray(bytearray(img_f.read()), dtype=np.uint8)
        img_f = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        mask = None
        leaf_mask = None

        with col1:
            st.image(img_f, channels="BGR", caption="Uploaded Image", width=200)

        plt.figure(figsize=(8, 5))

        kernel_sizes = list(range(3, 21, 2))
        results = []
        for k in kernel_sizes:
            try:
                res = calc_area(img_f, kernel_size=k, ref_size=ref_size)
                results.append(res[0])
                if (k==5):
                    leaf_mask = res[1]
                    mask = res[2]

            except Exception as e:
                print(f"Error with {img_f}, kernel={k}: {e}")
                results.append(0)

        with col2:
            st.image(mask, caption="Reference Square", width=200)
        with col3:
            st.image(leaf_mask, caption="Leaf Mask", width=200)
        df = pd.DataFrame({
            "Kernel Size": kernel_sizes,
            "Leaf Area": [f"{r:.2f}" for r in results]  # format numbers to 2 decimals
        })
        st.dataframe(df)  # allows sorting, scrolling, resizing

        plt.plot(kernel_sizes, results, marker='o')

        # Finalize combined plot
        plt.title('Effect of Morphological Kernel Size on Measured Leaf Area')
        plt.xlabel('Kernel Size (pixels)')
        plt.ylabel('Relative Leaf Area')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
