echo "Starting training..."
# on cloud:
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python train_td3.py &
# on macbook for debugging:
#python python train_td3.py&
sleep 1.0