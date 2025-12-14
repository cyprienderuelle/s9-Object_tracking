import cv2
import numpy as np
from KalmanFilter import KalmanFilter


def detect(frame):
    """
    Détection simple de cercles avec Hough
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=50
    )

    centers = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0]:
            centers.append((c[0], c[1]))

    return centers


def main_tracking():
    # filtre de Kalman
    kf = KalmanFilter(
        dt=0.1,
        u_x=1,
        u_y=1,
        std_acc=1,
        x_sdt_meas=0.1,
        y_sdt_meas=0.1
    )

    path = []

    cap = cv2.VideoCapture("randomball.avi")
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la vidéo")
        return

    # Récupérer les dimensions et FPS de la vidéo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialiser le VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('tracking_results/tp1.avi', fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        centers = detect(frame)

        detected = None
        if len(centers) > 0:
            detected = centers[0]
            x_d, y_d = detected
            cv2.circle(frame, (x_d, y_d), 20, (0, 255, 0), 2)

        # prédiction
        pred = kf.predict()
        x_p, y_p = int(pred[0]), int(pred[1])
        cv2.rectangle(
            frame,
            (x_p - 15, y_p - 15),
            (x_p + 15, y_p + 15),
            (255, 0, 0),
            2
        )

        # correction si détection
        if detected is not None:
            z = np.array([[x_d], [y_d]])
            est = kf.update(z)

            x_e, y_e = int(est[0]), int(est[1])
            path.append((x_e, y_e))

            cv2.rectangle(
                frame,
                (x_e - 15, y_e - 15),
                (x_e + 15, y_e + 15),
                (0, 0, 255),
                2
            )

            cv2.putText(frame, "Detected", (x_d + 20, y_d),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "Predicted", (x_p + 20, y_p - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Estimated", (x_e + 20, y_e + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # affichage de la trajectoire
        for i in range(1, len(path)):
            cv2.line(frame, path[i - 1], path[i], (0, 255, 255), 2)

        # Écrire la frame dans la vidéo de sortie
        out.write(frame)

        cv2.imshow("Kalman Tracking", frame)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_tracking()
