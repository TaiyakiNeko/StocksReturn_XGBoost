#!/bin/bash

echo "=============================================="
echo "  Stock Return Prediction Pipeline"
echo "=============================================="
echo ""

# Step 1: Train
echo "[Step 1/3] Training Model..."
echo "----------------------------------------------"
python train.py
if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi
echo ""

# Step 2: Inference
echo "[Step 2/3] Running Inference..."
echo "----------------------------------------------"
python main.py
if [ $? -ne 0 ]; then
    echo "❌ Inference failed!"
    exit 1
fi
echo ""

# Step 3: Evaluation
echo "[Step 3/3] Evaluating Results..."
echo "----------------------------------------------"
python evaluation.py
if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed!"
    exit 1
fi
echo ""

echo "=============================================="
echo "  Pipeline Complete!"
echo "=============================================="