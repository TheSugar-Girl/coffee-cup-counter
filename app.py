"""
Автоматизация учёта в кофейне — подсчёт стаканов/кружек
Стек: Streamlit + YOLOv8 (ultralytics) + OpenCV + fpdf2
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import json
import os
from datetime import datetime
from fpdf import FPDF
import tempfile
import io
import time

# ─── Константы ────────────────────────────────────────────────────────────────
HISTORY_FILE = "history.json"
CUP_CLASS_ID = 41          # COCO class: "cup"
MODEL_NAME   = "yolov8n.pt" # nano — быстрая, не требует GPU
CONF_THRESH  = 0.35

# ─── Загрузка модели (кешируется между сессиями) ───────────────────────────────
@st.cache_resource
def load_model():
    return YOLO(MODEL_NAME)

# ─── Детекция ─────────────────────────────────────────────────────────────────
def detect(model, img_np: np.ndarray):
    """Запускает YOLOv8, возвращает annotated-кадр и метрики."""
    results = model(img_np, classes=[CUP_CLASS_ID], conf=CONF_THRESH, verbose=False)
    r = results[0]
    annotated = r.plot()                          # BGR с bounding-boxes
    confs = r.boxes.conf.cpu().tolist() if r.boxes is not None and len(r.boxes) > 0 else []
    count = len(confs)
    avg_conf = round(float(np.mean(confs)), 3) if confs else 0.0
    return annotated, count, avg_conf

# ─── История ──────────────────────────────────────────────────────────────────
def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def append_history(source: str, count: int, avg_conf: float) -> list:
    history = load_history()
    history.append({
        "id":        len(history) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source":    source,
        "count":     count,
        "avg_conf":  avg_conf,
    })
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return history

def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

# ─── PDF-отчёт ────────────────────────────────────────────────────────────────
def generate_pdf(history: list) -> bytes:
    pdf = FPDF()
    pdf.add_page()

    # Заголовок
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "Coffee Shop Cup Detection Report", ln=True, align="C")
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.cell(0, 6, f"Model: {MODEL_NAME}  |  Confidence threshold: {CONF_THRESH}", ln=True, align="C")
    pdf.ln(6)

    # Сводка
    total_cups = sum(e["count"] for e in history)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, f"Total requests: {len(history)}   |   Total cups detected: {total_cups}", ln=True)
    pdf.ln(4)

    # Таблица
    pdf.set_fill_color(30, 30, 30)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 10)
    col_w = [12, 52, 36, 22, 30]
    headers = ["#", "Timestamp", "Source", "Cups", "Avg Conf"]
    for w, h in zip(col_w, headers):
        pdf.cell(w, 8, h, border=1, fill=True)
    pdf.ln()

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=9)
    for i, e in enumerate(history):
        fill = i % 2 == 0
        pdf.set_fill_color(245, 245, 245) if fill else pdf.set_fill_color(255, 255, 255)
        row = [str(e["id"]), e["timestamp"], e["source"][:30], str(e["count"]), str(e["avg_conf"])]
        for w, val in zip(col_w, row):
            pdf.cell(w, 7, val, border=1, fill=True)
        pdf.ln()

    return bytes(pdf.output())

# ─── Вспомогательные конверторы ───────────────────────────────────────────────
def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def pil_to_np(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))

# ─── UI ───────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="☕ Coffee Cup Counter",
        page_icon="☕",
        layout="wide",
    )

    st.title("☕ Автоматизация учёта в кофейне")
    st.caption("Подсчёт стаканов и кружек с помощью YOLOv8 (предобученная модель COCO)")

    model = load_model()

    # ── Боковая панель ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Настройки")
        mode = st.radio("Источник", ["📷 Изображение", "🎥 Видео", "📸 Веб-камера"])
        conf_slider = st.slider("Порог уверенности", 0.1, 0.9, CONF_THRESH, 0.05)

        st.divider()
        st.header("📋 История")
        history = load_history()
        st.metric("Всего запросов", len(history))
        st.metric("Всего кружек", sum(e["count"] for e in history))

        if history:
            pdf_bytes = generate_pdf(history)
            st.download_button(
                "📥 Скачать PDF-отчёт",
                data=pdf_bytes,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
            )
            hist_json = json.dumps(history, ensure_ascii=False, indent=2)
            st.download_button(
                "📥 Скачать JSON",
                data=hist_json.encode("utf-8"),
                file_name="history.json",
                mime="application/json",
            )
            if st.button("🗑️ Очистить историю"):
                clear_history()
                st.rerun()

    # ── Основной контент ────────────────────────────────────────────────────────

    # ── Режим: Изображение ──────────────────────────────────────────────────────
    if mode == "📷 Изображение":
        uploaded = st.file_uploader("Загрузи изображение", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            pil_img = Image.open(uploaded)
            col1, col2 = st.columns(2)
            col1.subheader("Оригинал")
            col1.image(pil_img, use_container_width=True)

            if st.button("🔍 Запустить детекцию"):
                with st.spinner("Обрабатываю..."):
                    img_np = pil_to_np(pil_img)
                    annotated, count, avg_conf = detect(model, img_np)
                    append_history(uploaded.name, count, avg_conf)

                col2.subheader(f"Результат — найдено: {count} 🥤")
                col2.image(bgr_to_rgb(annotated), use_container_width=True)
                st.success(f"✅ Обнаружено **{count}** стакан(ов)/кружек. Средняя уверенность: **{avg_conf:.2%}**")

    # ── Режим: Видео ────────────────────────────────────────────────────────────
    elif mode == "🎥 Видео":
        uploaded_video = st.file_uploader("Загрузи видео", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.flush()

            if st.button("🔍 Запустить детекцию на видео"):
                cap = cv2.VideoCapture(tfile.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                step = max(1, int(fps // 2))  # обрабатываем каждые ~0.5 сек

                stframe   = st.empty()
                prog_bar  = st.progress(0)
                stat_box  = st.empty()

                all_counts = []
                frame_idx  = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % step == 0:
                        annotated, count, avg_conf = detect(model, frame)
                        all_counts.append(count)
                        stframe.image(bgr_to_rgb(annotated), use_container_width=True, caption=f"Кадр {frame_idx} | Кружек: {count}")
                        stat_box.info(f"Средний подсчёт по кадрам: **{np.mean(all_counts):.1f}**")
                    prog_bar.progress(min(frame_idx / max(total_frames, 1), 1.0))
                    frame_idx += 1

                cap.release()
                avg_video = round(float(np.mean(all_counts)), 2) if all_counts else 0
                append_history(uploaded_video.name, int(avg_video), 0.0)
                st.success(f"✅ Видео обработано. Среднее кол-во кружек на кадр: **{avg_video}**")

    # ── Режим: Веб-камера ───────────────────────────────────────────────────────
    elif mode == "📸 Веб-камера":
        st.info("Сделай снимок с камеры, затем нажми «Детектировать»")
        cam_img = st.camera_input("Снимок с камеры")
        if cam_img:
            pil_img = Image.open(cam_img)
            if st.button("🔍 Детектировать"):
                with st.spinner("Анализирую снимок..."):
                    img_np = pil_to_np(pil_img)
                    annotated, count, avg_conf = detect(model, img_np)
                    append_history("webcam", count, avg_conf)

                col1, col2 = st.columns(2)
                col1.subheader("Снимок")
                col1.image(pil_img, use_container_width=True)
                col2.subheader(f"Найдено: {count} 🥤")
                col2.image(bgr_to_rgb(annotated), use_container_width=True)
                st.success(f"✅ Обнаружено **{count}** стакан(ов)/кружек. Средняя уверенность: **{avg_conf:.2%}**")

    # ── Таблица истории ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 История запросов")
    history = load_history()
    if history:
        st.dataframe(history, use_container_width=True)
    else:
        st.write("История пуста — запусти хотя бы одну детекцию.")


if __name__ == "__main__":
    main()