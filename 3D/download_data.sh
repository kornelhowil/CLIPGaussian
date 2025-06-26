FILEID="1OsiBs2udl32-1CqTXCitmov4NQCYdA9g"
FILENAME="nerf_sythetic.zip"
mkdir -p data

wget --no-check-certificate \
     "https://drive.usercontent.google.com/download?id=${FILEID}&confirm=t" \
     -O "data/${FILENAME}"

cd data
unzip ${FILENAME}
rm ${FILENAME}