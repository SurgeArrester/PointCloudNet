source activate pytorch

classifications=(OneMolecule TwoMolecules ThreeMolecules FourMolecules SixMolecules EightMolecules TwelveMolecules SixteenMolecules)

python train_classification.py --workers=1 --nepoch=25 --scoresFolder="output/$1/classification" --batchSize=22

for item in ${classifications[*]}
do
    python train_segmentation.py --workers=1 --nepoch=25 --batchSize=12 --scoresFolder="output/$1/segmentation/$item" --classification=$item
done
