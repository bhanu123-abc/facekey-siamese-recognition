# FaceKey — browser face verification

This project verifies one enrolled person using the existing Siamese model. It is a local browser application: webcam frames are sent only to the local Python server.

> The trained model (`siamesemodel.h5`), training checkpoints, and enrollment images are deliberately excluded from this public repository because they are large and may contain private biometric data. Place your own compatible model file in the project root before running the app.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5055 and allow camera access.

## First-use checklist

1. Click **Reset reference set** to remove any old or mixed-identity enrollment photos.
2. Capture all 12 samples of only the person the app should recognise. Use neutral, slight left/right, and different well-lit angles.
3. Verify in similarly good light. Frames with no face, a small face, blur, or darkness are rejected before model scoring.
4. If good samples still fail, collect clean positive samples and images of other people in `data/negative/`, then run `python -m ml.calibrate`.

The displayed match score is the Siamese model's similarity output, not a probability or a face-detection percentage. Re-training with correctly labelled same-person positive pairs and different-person negative pairs is still the proper remedy for a model trained on mixed identities.
