@echo off
REM This script runs a series of YOLO evaluation commands.

REM Check if the patch name argument is provided.
IF "%~1"=="" (
    echo Error: Missing patch name.
    echo Usage: %0 ^<patch_name_without_extension^>
    exit /b 1
)

REM Store the first argument in a variable.
set PATCH_NAME=%1

echo --- Starting Evaluation for Patch: %PATCH_NAME% ---

REM --- YOLOv5 Models ---
echo.
echo [INFO] Running YOLOv5 evaluations...
python evaluation.py --model yolov5n --patch ./patch/%PATCH_NAME%.png
python evaluation.py --model yolov5s --patch ./patch/%PATCH_NAME%.png
python evaluation.py --model yolov5m --patch ./patch/%PATCH_NAME%.png

REM --- YOLOv8 Models ---
echo.
echo [INFO] Running YOLOv8 evaluations...
python evaluation.py --model yolov8n --patch ./patch/%PATCH_NAME%.png
python evaluation.py --model yolov8s --patch ./patch/%PATCH_NAME%.png
python evaluation.py --model yolov8m --patch ./patch/%PATCH_NAME%.png

REM --- YOLOv9 Models ---
echo.
echo [INFO] Running YOLOv9 evaluations...
python evaluation.py --model yolov9t --patch ./patch/%PATCH_NAME%.png
python evaluation.py --model yolov9s --patch ./patch/%PATCH_NAME%.png
python evaluation.py --model yolov9m --patch ./patch/%PATCH_NAME%.png

REM --- YOLOv10 Models ---
echo.
echo [INFO] Running YOLOv10 evaluations...
python evaluation.py --model yolov10n --patch ./patch/%PATCH_NAME%.png
python evaluation.py --model yolov10s --patch ./patch/%PATCH_NAME%.png
python evaluation.py --model yolov10m --patch ./patch/%PATCH_NAME%.png

echo.
echo --- Evaluation complete. ---
