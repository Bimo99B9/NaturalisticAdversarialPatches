if [ -z "$1" ]; then
    echo "Error: Missing patch name"
    exit 1
fi

# python evaluation.py --model yolov4 --patch ./patch/$1.png

# python evaluation.py --model yolov4 --tiny --patch ./patch/$1.png

####
# python evaluation.py --model yolov5n --patch ./patch/$1.png
# python evaluation.py --model yolov5s --patch ./patch/$1.png
# python evaluation.py --model yolov5m --patch ./patch/$1.png

# python evaluation.py --model yolov8n --patch ./patch/$1.png
python evaluation.py --model yolov8s --patch ./patch/$1.png
python evaluation.py --model yolov8m --patch ./patch/$1.png

python evaluation.py --model yolov9t --patch ./patch/$1.png
python evaluation.py --model yolov9s --patch ./patch/$1.png
python evaluation.py --model yolov9m --patch ./patch/$1.png

python evaluation.py --model yolov10n --patch ./patch/$1.png
python evaluation.py --model yolov10s --patch ./patch/$1.png
python evaluation.py --model yolov10m --patch ./patch/$1.png
####

# python evaluation.py --model yolov4 --tiny --patch ./patch/v4.png
# python evaluation.py --model yolov4 --tiny --patch ./patch/v4tiny.png
# python evaluation.py --model yolov4 --tiny --patch ./patch/v4.png
# python evaluation.py --model yolov4 --tiny --patch ./patch/v4.png
# python evaluation.py --model yolov4 --tiny --patch ./patch/faster.png
# python evaluation.py --model yolov4 --tiny --patch ./patch/advyolo.png
# python evaluation.py --model yolov4 --tiny --patch ./patch/advyobjupper.png