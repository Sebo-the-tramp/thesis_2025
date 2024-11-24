echo "Preparing for 0.1fps"

cd campus_big_0.1fps

# mkdir -p 00
# cd 00
# mkdir ./images
# ffmpeg -hwaccel auto -i ../../GX010049.MP4 -vf "fps=0.1,scale=iw:ih" -q:v 2 ./images/%06d.jpg
# cd ..

# mkdir -p 01
# cd 01
# mkdir ./images
# ffmpeg -hwaccel auto -i ../../GX010044.MP4 -vf "fps=0.1,scale=iw:ih" -q:v 2 ./images/%06d.jpg
cd ../../
cp ../run.sh ./run.sh


# echo "Preparing for 1 fps"

# cd /campus/campus_big_1.0fps/

# cd 00
# mkdir ./images
# ffmpeg -hwaccel auto -i ../GX010049.MP4 -vf "fps=1,scale=iw:ih" -q:v 2 ./images/%06d.jpg
# cd ..

# cd 01
# mkdir ./images
# ffmpeg -hwaccel auto -i ../GX010044.MP4 -vf "fps=1,scale=iw:ih" -q:v 2 ./images/%06d.jpg
# cp ../../../run.sh ./run.sh
# cd ../../


# echo "Preparing for 5 fps"

# cd /campus/campus_big_5.0fps

# mkdir -p 00
# cd 00
# mkdir ./images
# ffmpeg -hwaccel auto -i ../GX010049.MP4 -vf "fps=5,scale=iw:ih" -q:v 2 ./images/%06d.jpg
# cd ..

# mkdir -p 01
# cd 01
# mkdir ./images
# ffmpeg -hwaccel auto -i ../GX010044.MP4 -vf "fps=5,scale=iw:ih" -q:v 2 ./images/%06d.jpg
# cp ../../../run.sh ./run.sh
# cd ../../


# I don't know how to do this any
# echo "Preparing for sparse fps" 

# cd /campus/campus_big_1.0fps/

# cd 00
# mkdir ./images
# ffmpeg -hwaccel auto -i ../GX010049.MP4 -vf "fps=1.0,scale=iw:ih" -q:v 2 ./images/%06d.jpg
# cd ..

# cd 01
# mkdir ./images
# ffmpeg -hwaccel auto -i ../GX010044.MP4 -vf "fps=1.0,scale=iw:ih" -q:v 2 ./images/%06d.jpg
# cp ../../../run.sh ./run.sh
# cd ..