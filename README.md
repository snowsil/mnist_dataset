# mnist_dataset
# data transform
A small program for processing the mnist data set, using the example network, with an accuracy of 99%.
After processing the edge detection of the data set, the accuracy is 98%.
The main program is modified on the basis of https://zhuanlan.zhihu.com/p/95701775.
The network adopts https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py.



# tools
some .sh and .py files that are used to transform dataset.

add_0.sh:change pictures' names from "0.jpg" to "000.jpg"("34.jpg" to "034.jpg").

count_no_anno:random sample 1000 videos in dataset. Count how many videos franmes without annos are over 50%.


divide_data:random sample part of dataset, print their filename in new .txt file.

divide_frame:original annos are for all videos, this file read origin files and write new file of each video, each frame on a row.

frame_2_video:result files are frames, transform them to a visible video.

move_files:refer to the src file, move files to some directory.

print:transform .txt directory file from "test/Apply_Eyebrow/000,128" to "test/Apply_Eyebrow_0"

print_content:transform .txt directory file from "test/Apply_Eyebrow/0,128" to "test/Apply_Eyebrow_0"

pth_2_txt:read .pth files, print some content to .txt file.

remove_bad_videos:move all videos whose frames without annos over 50%(bad videos) to other directory.

video_2_frames:divide a video to frames.
