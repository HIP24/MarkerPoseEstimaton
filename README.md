# MarkerPoseEstimaton

## Prerequisites

OpenCV version: 3.4.20


## Aufbau

Die Abgabe "MarkerPoseEstimation" besteht aus folgenden Teilen
- data
    - activeSet_XYZ.csv -> Abmessungen der Objekte
    - camera_params.yaml -> Intrinsische Kameraparameter
    - video.mp4 -> Video, das im Program verarbeitet wird

- output
    - activeSet_XYZ.csv -> Abmessungen der Objekte
    - camera_params.yaml -> Intrinsische Kameraparameter
    - video.mp4 -> Video, das im Program verarbeitet wird

- makefile -> Makefile für das Projekt

- markerPoseEstimation -> Das Programm

- prog -> ausführbare Datei (chmod +x prog)


## Ausführen des Programms

Um das Programm zu starten, muss es zuerst im Root kompilliert 
`make` 
Anschließend kann es ausgeführt werden
`./prog`. 


## Video

Eine 