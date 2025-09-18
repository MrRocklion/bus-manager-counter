#version 2
import cv2
import requests
import numpy as np
import degirum as dg
import degirum_tools
from concurrent.futures import ThreadPoolExecutor
import argparse


# funciones
def get_config_device() -> dict | None:
    """Obtiene la configuración del dispositivo desde un archivo JSON."""
    try:
        response = requests.get("http://localhost:8000/api/counter_configs/last")
        response.raise_for_status()
        aux_response = response.json()
        return aux_response['result']
    except requests.RequestException as e:
        print(f"Error al obtener la configuración del dispositivo: {e}")
        return None
    

init_data = get_config_device()
print(init_data)

zones = init_data['excluded_areas']
formated_zones = []
for zone in zones:
    aux_polygon = []
    for polygon in zone:
        aux_polygon.append([polygon['x'], polygon['y']])
    formated_zones.append(aux_polygon)

# Configuraciones
VIDEO_PATH = f"rtsp://{init_data['user_camera']}:{init_data['password_camera']}@{init_data['ip_counter_camera']}:554/cam/realmonitor?channel=1&subtype=0"
CLASS_LIST = ["Person"]
CROSS_LINE_Y = init_data['cross_line_y']
MODEL_PATH = "/home/admin/bus-manager-counter/yolov8n_relu6_person--640x640_quant_hailort_multidevice_1"
MODEL_NAME = "yolov8n_relu6_person--640x640_quant_hailort_multidevice_1"
API_URL = "http://localhost:8000/api/passengers"
TRACK_BUFFER = init_data['track_buffer']
TRACK_TRESH = init_data['track_threshold']


def get_bottom_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, y2


def send_api_request(special=False):
    try:
        r = requests.post(API_URL, headers={"Content-Type": "application/json"}, json={"special": special}, timeout=3)
        print("API OK:" if r.ok else f"API error: {r.status_code}", r.json() if r.ok else "")
    except requests.RequestException as e:
        print("API conexión fallida:", e)


def mask_frame_generator(video_source, cross_line_y=550, zones=[]):
    stream = cv2.VideoCapture(video_source)
    try:
        while True:
            ret, frame = stream.read()
            if not ret:
                break
            height, width, _ = frame.shape
            mask = np.ones((height, width), dtype=np.uint8) * 255

            excluded_zones = []
            for zone in zones:
                excluded_zones.append(np.array(zone, dtype=np.int32))
            cv2.fillPoly(mask, excluded_zones, 0)
            
            mask_3ch = cv2.merge([mask, mask, mask])
            frame = cv2.bitwise_and(frame, mask_3ch)
            cv2.line(frame, (0, cross_line_y), (width, cross_line_y), (0, 255, 0), 2)

            yield frame
    finally:
        print("Closing stream")
        stream.release()


def main(show_window=False):
    model = dg.load_model(
        model_name=MODEL_NAME,
        inference_host_address="@local",
        zoo_url=MODEL_PATH,
        token='',
        overlay_color=[(255, 0, 0)],
        output_class_set=set(CLASS_LIST)
    )

    tracker = degirum_tools.ObjectTracker(
        class_list=CLASS_LIST,
        track_thresh=TRACK_TRESH,
        track_buffer=TRACK_BUFFER,
        match_thresh=0.85,
        trail_depth=15,
        show_overlay=False,
        anchor_point=degirum_tools.AnchorPoint.BOTTOM_CENTER
    )

    degirum_tools.attach_analyzers(model, [tracker])
    counter = 0
    crossed_ids = set()

    with ThreadPoolExecutor(max_workers=5) as pool:
        for result in model.predict_batch(mask_frame_generator(VIDEO_PATH, cross_line_y=CROSS_LINE_Y, zones=formated_zones)):
            if show_window:
                cv2.imshow("CAM COUNTER", result.image_overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            for track_id, trail in result.trails.items():
                if len(trail) >= 2:
                    prev_x, prev_y = get_bottom_center(trail[-2])
                    curr_x, curr_y = get_bottom_center(trail[-1])
                    if prev_y < CROSS_LINE_Y <= curr_y and track_id not in crossed_ids:
                        crossed_ids.add(track_id)
                        counter += 1
                        print(f"Track ID {track_id} cruzó hacia abajo. Total: {counter}")
                        pool.submit(send_api_request, False)

    if show_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contador de personas con cámara RTSP y modelo DeGirum")
    parser.add_argument(
        "--show",
        type=str,
        default="false",
        help="Mostrar ventana con overlay (true/false). Por defecto: false"
    )
    args = parser.parse_args()

    # Convertir a booleano
    show_window = args.show.lower() in ["true", "1", "yes", "y"]

    main(show_window=show_window)
