cd /campus/section_01

cd 00
mkdir ./images
ffmpeg -hwaccel auto -i ./video/GX010041.MP4 -vf "fps=4,scale=iw:ih" -q:v 8 ./images/%06d.png
cd ..

cd 01
mkdir ./images
ffmpeg -hwaccel auto -i ./video/GX010036.MP4 -vf "fps=4,scale=iw:ih" -q:v 8 ./images/%06d.png
cp ../../run.sh ./run.sh
cd ..