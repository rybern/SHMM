OUT=test/test_posterior.csv
rm $OUT

# example from https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
./shmm test/trans.st test/emissions.csv $OUT
cat $OUT
echo ""
echo posterior error:
diff $OUT test/test_groundtruth.csv
