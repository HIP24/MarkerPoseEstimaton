# MarkerPoseEstimaton

## Prerequisites

OpenCV version: 3.4.20


## Aufbau

Die Abgabe "MarkerPoseEstimation" besteht aus folgenden Teilen
- data
    - activeSet_XYZ.csv -> Abmessungen der Objekte
    - camera_params.yaml -> Intrinsische Kameraparameter
    - video.mp4 -> Video, das im Program verarbeitet wird

- makefile -> Makefile für das Projekt

- markerPoseEstimation -> Das Programm

- prog -> Ausführbare Datei (chmod +x prog)

- output
    - graph_solvePnP.png -> Visualisierung vom Programmablauf mit solvePnP
    - graph_solvePnP.xlsx -> Excel Tabelle vom Programmablauf mit solvePnP
    - graph_RANSAC_100.png -> Visualisierung vom Programmablauf mit 100 Iterationen bei RANSAC
    - graph_RANSAC_100.xlsx -> Excel Tabelle vom Programmablauf mit 100 Iterationen bei RANSAC
    - graph_RANSAC_1000.png -> Visualisierung vom Programmablauf mit 1000 Iterationen bei RANSAC
    - graph_RANSAC_1000.xlsx -> Excel Tabelle vom Programmablauf mit 1000 Iterationen bei RANSAC
    - graph_error_100.png -> Visualisierung des Fehlers von RANSAC zu solvePnP bei 100 Iterationen
    - graph_error_100.xlsx -> Excel Tabelle des Fehlers von RANSAC zu solvePnP bei 100 Iterationen
    - graph_error_1000.png -> Visualisierung des Fehlers von RANSAC zu solvePnP bei 1000 Iterationen
    - graph_error_1000.xlsx -> Excel Tabelle des Fehlers von RANSAC zu solvePnP bei 1000 Iterationen
    - video_output.mp4 -> Beispielablauf für das Programm

## Ausführen des Programms

Um das Programm zu starten, muss es zuerst im Root kompilliert werden mit `make`. Anschließend kann es ausgeführt werden mit `./prog`. 

Ein Beispiel des Ablaufs wird im output Ordner bei video_output.mp4 gezeigt
