# example from https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
./shmm test/trans.st test/emissions.csv test/test_posterior.csv
cat test/test_posterior.csv
echo ""
echo posterior error:
diff test/test_posterior.csv test/test_groundtruth.csv
