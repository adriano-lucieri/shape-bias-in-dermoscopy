#!/bin/bash
set -e -o pipefail

echo "=================================================="
echo "Start"
echo "=================================================="

TASK=$1 # can be show to show the selected settings

cd ../scripts/
ROOT=../../
DATASETROOT=../../data
CONFIGBASEPATH=$ROOT/src/configs/EXP05/

SETUP=0

CONFIG_FILES=(
    #
    00_Derm7pt-b/08_Derm7pt-b_APRP.ini
    00_Derm7pt-b/09_Derm7pt-b_AF-APRP.ini
    00_Derm7pt-b/10_Derm7pt-b_MIX-APRP.ini
    #
    01_Derm7pt/08_Derm7pt_APRP.ini
    01_Derm7pt/09_Derm7pt_AF-APRP.ini
    01_Derm7pt/10_Derm7pt_MIX-APRP.ini
    #
    02_ISICb-NV-MEL/08_ISICb-NV-MEL_APRP.ini
    02_ISICb-NV-MEL/09_ISICb-NV-MEL_AF-APRP.ini
    02_ISICb-NV-MEL/10_ISICb-NV-MEL_MIX-APRP.ini
    #
    03_ISIC/08_ISIC_APRP.ini
    03_ISIC/09_ISIC_AF-APRP.ini
    03_ISIC/10_ISIC_MIX-APRP.ini
    #
    04_Imagenette/03_Imagenette_APRP.ini
    04_Imagenette/04_Imagenette_AF-APRP.ini
    04_Imagenette/05_Imagenette_MIX-APRP.ini
    #
    05_Imagewoof/03_Imagewoof_APRP.ini
    05_Imagewoof/04_Imagewoof_AF-APRP.ini
    05_Imagewoof/05_Imagewoof_MIX-APRP.ini
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
