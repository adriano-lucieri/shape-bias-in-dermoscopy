#!/bin/bash
set -e -o pipefail

echo "=================================================="
echo "Start"
echo "=================================================="

TASK=$1 # can be show to show the selected settings

cd ../scripts/
ROOT=../../
DATASETROOT=../../data
CONFIGBASEPATH=$ROOT/src/configs/EXP04/

SETUP=0

CONFIG_FILES=(
    00_Derm7pt-b/06_Derm7pt-b_replace-phase.ini
    00_Derm7pt-b/07_Derm7pt-b_replace-amplitude.ini
    #
    01_Derm7pt/06_Derm7pt_replace-phase.ini
    01_Derm7pt/07_Derm7pt_replace-amplitude.ini
    #
    02_ISICb-NV-MEL/06_ISICb-NV-MEL_replace-phase.ini
    02_ISICb-NV-MEL/07_ISICb-NV-MEL_replace-amplitude.ini
    #
    03_ISIC/06_ISIC_replace-phase.ini
    03_ISIC/07_ISIC_replace-amplitude.ini
    #
    04_Imagenette/00_Imagenette_baseline.ini
    04_Imagenette/01_Imagenette_replace-phase.ini
    04_Imagenette/02_Imagenette_replace-amplitude.ini
    #
    05_Imagewoof/00_Imagewoof_baseline.ini
    05_Imagewoof/01_Imagewoof_replace-phase.ini
    05_Imagewoof/02_Imagewoof_replace-amplitude.ini
)

for CID in ${!CONFIG_FILES[@]}
do
    echo "=================================================="
    echo "Setup: $CID Config: $BASEPATH${CONFIG_FILES[$CID]}" 
    echo "=================================================="

    if [[ $TASK != "show" ]]
    then
        python main.py --config $CONFIGBASEPATH${CONFIG_FILES[$CID]} -d $DATASETROOT -n 10
    fi
done

echo "=================================================="
echo "Finished execution"
echo "=================================================="
