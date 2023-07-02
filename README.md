# MarkerPoseEstimaton

## Voraussetzungen

OpenCV version: 3.4.20


## Aufbau

Die Abgabe "MarkerPoseEstimation" besteht aus folgenden Teilen
- data
    - activeSet_XYZ.csv -> Abmessungen der Objekte
    - camera_params.yaml -> Intrinsische Kameraparameter
    - video.mp4 -> Video, das im Program verarbeitet wird

- makefile -> Makefile für das Projekt

- markerPoseEstimation.cpp -> Das Programm

- prog -> Ausführbare Datei (chmod +x prog)

- output
    - graph_solvePnP.png -> Visualisierung vom Programmablauf mit solvePnP
    - graph_solvePnP.xlsx -> Excel Tabelle vom Programmablauf mit solvePnP
    - graph_RANSAC_100_8.png -> Visualisierung vom Programmablauf mit 100 Iterationen und 8 Reprojektionsfehlern bei RANSAC
    - graph_RANSAC_100_8.xlsx -> Excel Tabelle vom Programmablauf mit 100 Iterationen und 8 Reprojektionsfehlern bei RANSAC
    - graph_RANSAC_100_8_adjusted.png -> Visualisierung vom Programmablauf mit 100 Iterationen und 8 Reprojektionsfehlern bei RANSAC mit angepassten Werten
    - graph_RANSAC_100_8_adjusted.xlsx -> Excel Tabelle vom Programmablauf mit 100 Iterationen und 8 Reprojektionsfehlern bei RANSAC mit angepassten Werten
    - graph_RANSAC_1000_2.png -> Visualisierung vom Programmablauf mit 1000 Iterationen und 2 Reprojektionsfehlern bei RANSAC
    - graph_RANSAC_1000_2.xlsx -> Excel Tabelle vom Programmablauf mit 1000 Iterationen und 2 Reprojektionsfehlern bei RANSAC
    - graph_error_100_8.png -> Visualisierung des Fehlers von RANSAC zu solvePnP bei 100 Iterationen und 8 Reprojektionsfehlern
    - graph_error_100_8.xlsx -> Excel Tabelle des Fehlers von RANSAC zu solvePnP bei 100 Iterationen und 8 Reprojektionsfehlern
    - graph_error_1000_2.png -> Visualisierung des Fehlers von RANSAC zu solvePnP bei 1000 Iterationen und 2 Reprojektionsfehlern
    - graph_error_1000_2.xlsx -> Excel Tabelle des Fehlers von RANSAC zu solvePnP bei 1000 Iterationen und 2 Reprojektionsfehlern
    - video_output.mp4 -> Beispielablauf für das Programm

## Ausführen des Programms

Um das Programm zu starten, muss es zuerst im Root kompilliert werden mit `make`. Anschließend kann es ausgeführt werden mit `./prog`. 

Ein Beispiel des Ablaufs wird im *output* Ordner bei *video_output.mp4* gezeigt.

## Nach dem Ausführen des Programms 

Nach dem Ausführen des Programms werden drei neue Excel Dateien erzeugt
- SolvePnP.csv -> tvec und rvec Werte von solveP
- RANSAC.csv -> tvec und rvec Werte von RANSAC
- Error.csv -> Unterschied der tvec und rvec Werte von solvePnP und RANSAC


