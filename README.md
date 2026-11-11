# Road Obstacle Detection

The program reads one fixed camera or recorded video, processes a 320 × 240 view at approximately five frames per second, and tracks moving regions through three measurement lines. It compares passage times over equal intervals in the rectified view. A possible pothole alert requires repeated slowdown in one lane and enough normal passages in the other. Three tracked merges from one lane within 20 seconds trigger urgent review. Both alerts save a small JPEG locally.

## Run from VS Code

Open this folder in VS Code and run the following in its terminal. The program uses OpenCV and NumPy, both available on the current laptop. For another Python environment, install them with `python3 -m pip install opencv-python numpy`.

```sh
python3 monitor.py 0
```

Press **Q** to close the preview and release the camera. Add `--headless --seconds 5` for a short camera check without a window. For an external webcam, check `ls -l /dev/v4l/by-id/` and use its numbered capture device, for example `2` for `/dev/video2`.

## Calibrate a real camera

Mount the camera so both lanes are visible and keep it fixed. This version expects the two lanes to carry traffic from the bottom toward the top of the rectified view. Set `--road` to the far-left, far-right, near-right and near-left road corners, expressed as x/y pairs in the 320 × 240 preview. Choose a short segment with straight lanes. The preview shows the lane divider and the three measurement lines. Use `--speed-breaker` at a known speed breaker to suppress pothole alerts.

The default pothole threshold is 100 distinct tracked passages over a rolling 72 hours in a single session. Calibrated live sessions restore this history from SQLite when using the desktop app or local dashboard. Moving regions can merge or split in dense traffic. Camera motion, shadows and people can also produce tracks. Use labeled footage to measure this before interpreting counts. An urgent alert currently checks merges and writes a JPEG. The desktop app can use the STM32 decision firmware over USB. A physical buzzer is outside this prototype.

## Evaluation data

Use [highD](https://levelxdata.com/highd-dataset/) for real vehicle trajectories and lane-change tests, subject to its access terms. Use [RDD2022](https://crddc2022.sekilab.global/data/) for labeled pothole images. Those images do not provide vehicle trajectories. For end-to-end evaluation, use fixed two-lane videos with timestamps and human-labeled speed breakers, lane obstructions and possible pothole stops. Keep test sites separate from the sites used to tune thresholds.

## Desktop screen and USB controller

From this folder in VS Code:

```sh
python3 app.py
```

**Start camera** defaults to the built-in camera. **Open video** replays a local file.

Firmware is in `firmware/`. It builds with GNU Arm Embedded GCC via `make -C firmware`. After programming and verifying `firmware/hazard.elf` on the NUCLEO-F103RB with ST-LINK, use:

```sh
python3 app.py --port /dev/ttyACM0
```

The laptop extracts passages and keeps the recent history. The board decides whether the counts warrant an alert and lights onboard LD2 when it returns an alert. The application validates the board reply before displaying a result in USB mode. Urgent alerts also request the desktop system bell. Replay and live preview use the same tracker.

The real-data smoke test used the stabilized [Kumasi pothole clip](https://commons.wikimedia.org/wiki/File:Cars_Moving_into_pothole.webm) by Amuzujoe, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The first motion-only run yielded zero completed passages.

## Real traffic replay

Select **Real traffic replay** for a 33-second highway recording played at its original speed. The app loads the calibration for the two rightmost lanes. This checks vehicle passage tracking and normal traffic behavior. The sample was downloaded for local testing from [Chris Dahms’s CarsDrivingUnderBridge.mp4](https://github.com/MicrocontrollersAndMore/OpenCV_3_Car_Counting_Cpp/blob/master/CarsDrivingUnderBridge.mp4). Verify footage rights before redistributing it.

Select **Real pothole replay** to watch 21 seconds of real traffic passing damaged pavement in Kumasi. The clip was stabilized and retains its 960 × 540 aspect ratio and all 311 frames. The replay uses a 320-pixel YOLO11n vehicle detector exported to ONNX, with measurement lines calibrated for the visible approach. The retest measured three passages and two slowdowns. This falls below the repeated-slowdown alert threshold. The ONNX weights are stored locally at data/vehicle-detector.onnx. The regular camera and highway modes still use the motion tracker. Footage by Amuzujoe, adapted from [Cars Moving into pothole](https://commons.wikimedia.org/wiki/File:Cars_Moving_into_pothole.webm) under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

### Event-triggered pavement suggestion

In **Real pothole replay**, the first measured slowdown opens a short image-check window. The laptop loads the local [pothole-trained ONNX model](https://huggingface.co/tahaUgan/pothole-yolo11n/blob/main/pothole2v_320.onnx) on demand and samples at most 12 low-resolution frames per check. A suggestion needs overlapping detections in two sampled frames. After a suggestion, the run saves one annotated JPEG and stops pavement inference. Marking the site as a known speed breaker disables this check. The image suggestion and STM32 traffic alert decision appear independently in the UI.

### Lane obstruction replay and sound

**Real blockage replay** loads the complete TAD-benchmark test clip T9 from the [AI City Challenge Track 3 dataset](https://huggingface.co/datasets/Tohrumi/AICityChallenge_Track_3). The app reads the entire recording. The existing lane calibration applies to the first 15 seconds. Vehicle detection continues over the full image after the pan. The prototype counts tracked merges within the calibrated road segment. The STM32 requires three merges out of the same lane within 20 seconds for an urgent review. An urgent event displays its source image and plays a short two-tone WAV through PipeWire or PulseAudio, with the desktop bell as a fallback.

Highway obstruction and Kumasi vehicle tracking use a 320-pixel YOLO11n ONNX export through OpenCV DNN. Pavement suggestions use the separate pothole-trained model described above. The lane blockage decision is a rule evaluated on the STM32 in USB mode.

For **Open video** and **Start camera**, the app detects vehicles across the image by default and preserves the source aspect ratio in the preview. Pass camera-specific `--road` corners to enable lane alerts. Videos run to end of file, and live analysis continues until **Stop** or the device disconnects. The app reads the source and samples about five frames per second for inference.

The replay previews are scaled directly from the widescreen recordings. The tracker still uses a small working image for inference. The pothole replay retains 311 frames at 960 × 540 and the blockage replay retains 499 frames at 1920 × 1080.

The blockage replay also checks a calibrated active-lane crop twice a second with the same vehicle model. A vehicle that creeps sideways with little forward movement for at least three seconds can trigger a visual blockage review. The STM32 accepts this evidence through the USB options byte, where bit 0 marks a known speed breaker and bit 1 marks a visual blockage candidate. This check is configured for the fixed view in the supplied blockage replay. It covers the demonstrated disabled-vehicle pattern. Other obstruction types and camera views require separate validation.

### Automatic speed-breaker replay

**Real speed breaker replay** uses the 35-second public stock preview [Cars and Vans Driving Over a Speed Bump](https://elements.envato.com/speed-bump-KQUCJ4D) by LuxuryFrame. The app starts with pothole suppression off. A compact [YOLOv5 speed-bump model](https://github.com/ikhlaq34/Potholes-and-Speed-Bumps-Detection-System) exported to ONNX checks sampled road crops. The prototype requires its prediction to overlap a textured strip in two consecutive samples before showing the evidence and sending the speed-breaker flag over USB. It samples at most 14 times, once per second, and stops on confirmation. This prototype is tuned for the visible modular rubber strip and calibrated replay. Other camera angles and bump types need separate validation.

## Local web dashboard and saved history

Run `python3 dashboard.py --port /dev/ttyACM0` and open `http://127.0.0.1:8765/`. The dashboard runs locally and uses the same USB controller and replay pipeline as the desktop app. Use **Sound** to enable alerts in the browser.

For persistent live counts, start the dashboard with `--camera 0 --road` followed by the eight calibrated road-corner coordinates in the 320 × 240 reference view. The dashboard restores the last 72 hours for that camera index and calibration. Calibrate each view before enabling lane alerts. Changing the camera index or road calibration selects a separate history. The desktop app uses the same database when started with matching calibration.

The SQLite file is `data/history.sqlite3`. Recorded videos save their alerts and image suggestions but never add to the live totals. Each live passage is committed as it completes. The dashboard displays the 40 most recent saved events. Evidence JPEGs are stored in `evidence/`.

In the browser, choose **Mark lanes on camera** and **Freeze view to mark lanes**. Click the far-left, far-right, near-right and near-left corners around both lanes, then **Save lanes**. The center divider is inferred from the outline. **Start monitoring** runs the camera and USB controller in the background. The browser shows counts, alerts and saved evidence. **Stop** releases the camera. Calibration stays in SQLite across restarts. The same camera index and corners restore the same rolling history.

**Auto-mark lanes** uses the local [YOLOP](https://github.com/hustvl/YOLOP) lane-line model for camera setup only. It looks for three stable markings defining two adjacent lanes. The automatic fit assumes traffic travels away from the camera through those lanes. It requires a fixed view with clear lane markings. Start monitoring reuses saved calibration, or tries automatic calibration if none exists. An inconclusive attempt leaves existing calibration in place. The live feed stays hidden during background analysis.

Recorded clips use native browser playback. The built-in replays have compact MP4 preview copies under `data/preview/` that preserve the source frame rate. Analysis continues separately at about five frames per second on the original input. **Show tracking view** enables annotated snapshots on demand. The live camera stays hidden during background monitoring.
