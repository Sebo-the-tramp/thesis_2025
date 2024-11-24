# RUN RECONSTRUCTION
cd all

echo "Running Feature Extractor"
colmap feature_extractor --database_path ./database.db --image_path ./images --ImageReader.single_camera_per_folder 1 --ImageReader.default_focal_length_factor 0.5 --ImageReader.camera_model OPENCV 

echo "Running feature matching..."
colmap sequential_matcher --database_path ./database.db --SiftMatching.max_distance 1 --SiftMatching.guided_matching 1 --SequentialMatching.overlap 2

echo "Reconstructing 3D model..."
colmap mapper --database_path ./database.db --image_path ./images --output_path ./

ENDIN=$(date +"%H:%M:%S")
curl -X POST -H 'Content-Type: application/json' -d "{\"chat_id\": \"777722458\", \"text\": \"FINE COLMAP\", \"disable_notification\": true}" https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage

echo "Showing result"
colmap gui --import_path ./0 --database_path ./database.db --image_path ./images