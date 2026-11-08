## PROJECT STRUCTURE

root
├── data-repo
│ ├── metadata
│ │ ├── labels.csv
│ │ ├── test-answers.csv
│ │ ├── test.csv
│ │ ├── train.csv
│ │ └── validation.csv
│ ├── target
│ │ ├── batch\_[i].npz # hold n batches of extracted data
│ │ ├── checkpoint.json
│ │ └── verification_report.json
│ └── raw-data
│ └── 20bn-jester-v1 # jester dataset
└── main
├── GUI
├── README.md
├── action.py
├── capture_video.py
├── gestures.py
├── hand_tracker.py
├── data
│ ├── process-data.py
│ └── verify-extracted-data.py
├── model
│ ├── gesture_recognizer.task
│ ├── hand_landmarker.task
│ └── training
│ └── training.py
├── requirements.txt
└── ui.py

## Notes

- current just main folder, which contains core code, is committed, data-repo is hold locally. The above project structure aims to show organization of data-repo and how main folder access the data.
