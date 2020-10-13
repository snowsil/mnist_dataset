#!/bin/bash:

cd test_data
path = test_data
for i in $(ls ${path})
do
	cd ${i}
	rename 's/^/00/' [0-9].jpg 
	rename 's/^/0/' [0-9][0-9].jpg
	cd ..
done

