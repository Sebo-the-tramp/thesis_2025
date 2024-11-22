rsync -av --ignore-existing --dry-run -e ssh sebastian.cavada@login-student-lab.mbzu.ae:/home/sebastian.cavada/Documents/scsv/thesis/hierarchical-3d-gaussians/_data ./_data

# ask if everything is good
echo "Do you want to proceed with the sync? (y/n)"
read -r response

if [ "$response" = "y" ]; then
    rsync -av --ignore-existing -e ssh sebastian.cavada@login-student-lab.mbzu.ae:/home/sebastian.cavada/Documents/scsv/thesis/hierarchical-3d-gaussians/_data ./_data
    echo "Sync completed."

else
    echo "Sync aborted."
fi