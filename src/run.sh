# This file runs the lane detection binary on the four clips available
# in the dataset

# Author: Mohamed Aly <malaa@caltech.edu>
# Date: 10/7/2010

#clips to run
path="../clips"
clips="cordova1 cordova2 washington1 washington2"

#get options
options=" --show --save-lanes --wait=50 --lanes-conf=Lanes.conf \
      --camera-conf=CameraInfo.conf "

# suffix
binary="./LaneDetector$(getconf LONG_BIT)"

#run
for clip in $clips; do
	echo "Running for $clip..."
  echo "------------------------------------------------------------------"
  echo

  # command
  command="$binary $options --list-file=$path/$clip/list.txt  \
    --list-path=$path/$clip/ --output-suffix=_results "
  echo $command

  # run
  $command
done
