""" Test module for Sifted stage to verify its functionality

The file contains multiple unit tests to ensure that the `Sifted` class corretly
perform its tasks. The basic mechanism of the test is to compare its output against
output from MATLAB and check if the outputs are the same or reasonable similar. The
tests also include some boundary test where appropriate to test the boundary of the
statement within the methods to ensure they are implemented appropriately.

Tests includes:
-  For the function select_features_by_performance, we check
    1. check if silhoulette value is in bell shape (?)
    2. correlation in descending order after filter out insignificant colleration values
    3. xaux value, check if features selected are the same based on correlation
- For the function costfcn, given an example, check if y value are same
- For the function ga, check if the filtered x value is the same (need to confirm with client)

"""

# cross correlation between matlab and pythoon, by row and by column, one and only one one corrlatino is above 0.9, others are lower than 0.3
# recommend best k value that gives the silhoulette value, and give warning if not bell value
# chang into int for binary matrix
# compare whether the faeture is in same cluster
# check error of machine learning model is wtithin a range
# line 133 in matlab, get [ind,y], cost value 